"""Map FC-CAS segments to proposal settings for a prior-art runtime carrier."""

from __future__ import annotations

from typing import Any

from fc_cas.strategy import StrategyTable
from fc_cas.types import SegmentInfo


def cas_proposal_hparams(segment: SegmentInfo, table: StrategyTable) -> dict[str, Any]:
    """Return proposal settings selected for one classified generation segment."""
    strategy = table.lookup(segment.region)
    return {
        "block_size": strategy.block_size,
        "confidence_threshold": strategy.confidence_threshold,
        "mode": strategy.mode.value,
        "allowed_strings": segment.allowed_strings,
    }
