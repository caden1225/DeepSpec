from __future__ import annotations

from fc_cas.types import DraftMode, Region, Strategy


DEFAULT_TABLE: dict[Region, Strategy] = {
    Region.NL: Strategy(block_size=4, confidence_threshold=0.6, mode=DraftMode.FREE),
    Region.TOOL_SKELETON: Strategy(
        block_size=24, confidence_threshold=0.2, mode=DraftMode.TEMPLATE
    ),
    Region.TOOL_ENUM: Strategy(
        block_size=16, confidence_threshold=0.2, mode=DraftMode.ENUM_MASK
    ),
    Region.TOOL_FREE: Strategy(
        block_size=8, confidence_threshold=0.4, mode=DraftMode.FREE
    ),
}


class StrategyTable:
    def __init__(self, table: dict[Region, Strategy] | None = None) -> None:
        self._table = dict(DEFAULT_TABLE if table is None else table)

    def lookup(self, region: Region) -> Strategy:
        try:
            return self._table[region]
        except KeyError as e:
            raise KeyError(f"no strategy for region={region!r}") from e

    def override(self, region: Region, strategy: Strategy) -> None:
        self._table[region] = strategy
