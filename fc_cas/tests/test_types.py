from fc_cas.types import DraftMode, Region, Strategy


def test_strategy_fields():
    s = Strategy(block_size=4, confidence_threshold=0.6, mode=DraftMode.FREE)
    assert s.block_size == 4
    assert Region.NL.value == "nl"
