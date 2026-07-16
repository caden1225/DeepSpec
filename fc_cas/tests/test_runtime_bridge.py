import importlib.util
from pathlib import Path

from fc_cas.strategy import StrategyTable
from fc_cas.types import DraftMode, Region, SegmentInfo


def test_cas_proposal_hparams_uses_segment_strategy_and_enum_constraint():
    from fc_cas.runtime.dspark_bridge import cas_proposal_hparams

    hparams = cas_proposal_hparams(
        SegmentInfo(
            region=Region.TOOL_ENUM,
            allowed_strings=("eco", "sport"),
            json_pointer="/arguments/mode",
        ),
        StrategyTable(),
    )

    assert hparams == {
        "block_size": 16,
        "confidence_threshold": 0.2,
        "mode": DraftMode.ENUM_MASK.value,
        "allowed_strings": ("eco", "sport"),
    }


def test_mock_compare_evaluates_twenty_test_samples_and_writes_results(tmp_path):
    script = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "scripts"
        / "run_compare_eval.py"
    )
    spec = importlib.util.spec_from_file_location("run_compare_eval", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run_mock_compare(
        data_root=Path(__file__).resolve().parents[1] / "data" / "processed_FC_dataset",
        split="test",
        limit=20,
        output_path=tmp_path / "compare_mock.json",
    )

    assert result["verify_mode"] == "mock_gold_next_token"
    assert result["samples_evaluated"] == 20
    assert result["baseline"]["accepted_characters"] > 0
    assert result["cas"]["accepted_characters"] > 0
    assert (tmp_path / "compare_mock.json").is_file()
