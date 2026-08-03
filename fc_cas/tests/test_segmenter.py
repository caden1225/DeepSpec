import json
from pathlib import Path

from fc_cas.segmenter import classify_prefix
from fc_cas.types import Region


DATA = Path(__file__).resolve().parents[1] / "data" / "processed_FC_dataset" / "test.json"


def _ota_sample():
    for ex in json.loads(DATA.read_text()):
        for m in ex["messages"]:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                tc = m["tool_calls"][0]
                if tc["function"]["name"] == "ota_update":
                    return ex, tc
    raise AssertionError("no ota_update tool_call")


def _function_with_property_type(prop_type):
    for ex in json.loads(DATA.read_text()):
        for tool in ex["tools"]:
            fn = tool.get("function", {})
            for key, schema in (fn.get("parameters", {}).get("properties", {})).items():
                if schema.get("type") == prop_type:
                    return ex, fn["name"], key
    raise AssertionError(f"no {prop_type} parameter fixture")


def test_nl_without_tool_json():
    info = classify_prefix("抱歉，车机暂不支持此功能。", tools=[], active_fn=None)
    assert info.region == Region.NL


def _serialized_tool_call(tc):
    return json.dumps({"tool_calls": [tc]}, ensure_ascii=False, separators=(",", ":"))


def _through(text, marker):
    index = text.index(marker)
    return text[: index + len(marker)]


def test_ota_update_serialized_prefix_regions():
    ex, tc = _ota_sample()
    payload = _serialized_tool_call(tc)
    cases = [
        ("自然语言", "请为我检查更新", Region.NL),
        ("调用骨架", _through(payload, '"tool_calls":['), Region.TOOL_SKELETON),
        ("函数名", _through(payload, '"name":"ota'), Region.TOOL_SKELETON),
        ("参数键", _through(payload, '\\"action'), Region.TOOL_SKELETON),
        ("参数分隔符", _through(payload, '\\"action\\":'), Region.TOOL_SKELETON),
        ("枚举值", _through(payload, '\\"action\\": \\"download'), Region.TOOL_ENUM),
        ("枚举值完成", _through(payload, '\\"action\\": \\"download_update\\"'), Region.TOOL_SKELETON),
    ]

    for label, prefix, expected_region in cases:
        active_fn = None if label == "自然语言" else "ota_update"
        info = classify_prefix(prefix, tools=ex["tools"], active_fn=active_fn)
        assert info.region == expected_region
        if label == "枚举值":
            assert info.allowed_strings == (
                "check_update",
                "download_update",
                "install_update",
                "schedule_update",
                "view_changelog",
            )
            assert info.json_pointer == "/arguments/action"


def test_ota_update_free_value_region():
    ex, tc = _ota_sample()
    prefix = (
        '{"tool_calls":[{"function":{"name":"ota_update",'
        '"arguments":"{\\"schedule_time\\":\\"2026-07-16'
    )
    info = classify_prefix(prefix, tools=ex["tools"], active_fn=tc["function"]["name"])
    assert info.region == Region.TOOL_FREE
    assert info.json_pointer == "/arguments/schedule_time"


def test_qwen_xml_tool_call_regions():
    ex, tc = _ota_sample()
    tools = ex["tools"]
    assert classify_prefix("先查一下", tools=tools).region == Region.NL
    sk = classify_prefix("<tool_call>\n<function=ota_update>\n", tools=tools)
    assert sk.region == Region.TOOL_SKELETON
    enum_p = classify_prefix(
        "<tool_call>\n<function=ota_update>\n<parameter=action>\ncheck",
        tools=tools,
    )
    assert enum_p.region == Region.TOOL_ENUM
    assert "check_update" in (enum_p.allowed_strings or ())
    free_p = classify_prefix(
        "<tool_call>\n<function=ota_update>\n<parameter=schedule_time>\n2026",
        tools=tools,
    )
    assert free_p.region == Region.TOOL_FREE


def test_unfinished_nested_array_and_object_values_are_free():
    array_ex, array_fn, array_key = _function_with_property_type("array")
    array_prefix = (
        f'{{"tool_calls":[{{"function":{{"name":"{array_fn}",'
        f'"arguments":{{"{array_key}":["first",'
    )
    array_info = classify_prefix(array_prefix, tools=array_ex["tools"], active_fn=array_fn)
    assert array_info.region == Region.TOOL_FREE
    assert array_info.json_pointer == f"/arguments/{array_key}"

    object_fn = "object_tool"
    object_key = "config"
    object_tools = [
        {
            "type": "function",
            "function": {
                "name": object_fn,
                "parameters": {
                    "type": "object",
                    "properties": {object_key: {"type": "object"}},
                },
            },
        }
    ]
    object_prefix = (
        f'{{"tool_calls":[{{"function":{{"name":"{object_fn}",'
        f'"arguments":{{"{object_key}":{{"nested":"value",'
    )
    object_info = classify_prefix(object_prefix, tools=object_tools, active_fn=object_fn)
    assert object_info.region == Region.TOOL_FREE
    assert object_info.json_pointer == f"/arguments/{object_key}"
