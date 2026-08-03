from fc_cas.strategy import DEFAULT_TABLE, StrategyTable
from fc_cas.types import DraftMode, Region


def test_tool_more_aggressive_than_nl():
    table = StrategyTable()
    nl = table.lookup(Region.NL)
    sk = table.lookup(Region.TOOL_SKELETON)
    assert sk.block_size > nl.block_size
    assert sk.confidence_threshold < nl.confidence_threshold


def test_enum_mode():
    assert DEFAULT_TABLE[Region.TOOL_ENUM].mode == DraftMode.ENUM_MASK
