from fc_cas.eval.offline_stats import compute_offline_stats, serialize_assistant


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_mode",
            "parameters": {
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["eco", "sport"]}},
            },
        },
    }
]


def test_serialize_assistant_concatenates_content_and_canonical_tool_calls():
    message = {
        "role": "assistant",
        "content": "正在设置。",
        "tool_calls": [
            {
                "type": "function",
                "id": "call_1",
                "function": {"arguments": '{"mode":"eco"}', "name": "set_mode"},
            }
        ],
    }

    assert serialize_assistant(message) == (
        '正在设置。{"tool_calls":[{"id":"call_1","type":"function",'
        '"function":{"name":"set_mode","arguments":"{\\"mode\\":\\"eco\\"}"}}]}'
    )


def test_compute_offline_stats_reports_regions_coverage_and_sample_types():
    examples = [
        {
            "tools": TOOLS,
            "messages": [{"role": "assistant", "content": "抱歉，暂不支持。"}],
        },
        {
            "tools": TOOLS,
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "set_mode",
                                "arguments": '{"mode":"eco"}',
                            },
                        }
                    ],
                }
            ],
        },
    ]

    stats = compute_offline_stats(examples, split="test")

    assert stats["split"] == "test"
    assert stats["assistant_sample_counts"] == {"nl": 1, "tool": 1}
    assert stats["total_serialized_chars"] > 0
    assert sum(stats["region_char_counts"].values()) == stats["total_serialized_chars"]
    assert stats["region_char_counts"]["nl"] > 0
    assert stats["region_char_counts"]["tool_skeleton"] > 0
    assert stats["region_char_counts"]["tool_enum"] > 0
    assert stats["template_enum_coverable_char_share"] > 0


def test_compute_offline_stats_uses_active_function_for_later_tool_calls():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "set_temperature",
                "parameters": {
                    "type": "object",
                    "properties": {"degrees": {"type": "number"}},
                },
            },
        },
        *TOOLS,
    ]
    examples = [
        {
            "tools": tools,
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "set_temperature",
                                "arguments": '{"degrees":22}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "set_mode",
                                "arguments": '{"mode":"eco"}',
                            },
                        },
                    ],
                }
            ],
        }
    ]

    stats = compute_offline_stats(examples, split="test")

    assert stats["region_char_counts"]["tool_enum"] > 0
