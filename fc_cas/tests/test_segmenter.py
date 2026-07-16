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


def test_ota_update_free_value_region():
    ex, tc = _ota_sample()
    prefix = (
        '{"tool_calls":[{"function":{"name":"ota_update",'
        '"arguments":"{\\"schedule_time\\":\\"2026-07-16'
    )
    info = classify_prefix(prefix, tools=ex["tools"], active_fn=tc["function"]["name"])
    assert info.region == Region.TOOL_FREE
    assert info.json_pointer == "/arguments/schedule_time"
