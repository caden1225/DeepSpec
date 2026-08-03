from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Region(str, Enum):
    NL = "nl"
    TOOL_SKELETON = "tool_skeleton"
    TOOL_ENUM = "tool_enum"
    TOOL_FREE = "tool_free"


class DraftMode(str, Enum):
    FREE = "free"
    TEMPLATE = "template"
    ENUM_MASK = "enum_mask"


@dataclass(frozen=True)
class Strategy:
    block_size: int
    confidence_threshold: float
    mode: DraftMode


@dataclass(frozen=True)
class SegmentInfo:
    region: Region
    allowed_strings: tuple[str, ...] | None = None
    json_pointer: str | None = None
