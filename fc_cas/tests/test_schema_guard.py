import json
from pathlib import Path

from fc_cas.schema import enum_allowed


DATA = Path(__file__).resolve().parents[1] / "data" / "processed_FC_dataset" / "test.json"


def test_ota_action_enum_from_dataset():
    samples = json.loads(DATA.read_text())
    # find first ota_update sample
    for ex in samples:
        names = [t["function"]["name"] for t in ex["tools"]]
        if "ota_update" in names:
            allowed = enum_allowed(ex["tools"], "ota_update", "action")
            assert allowed is not None
            assert "check_update" in allowed
            assert "download_update" in allowed
            return
    raise AssertionError("no ota_update sample in test split")
