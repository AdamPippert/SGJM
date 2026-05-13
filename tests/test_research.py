import pytest

torch = pytest.importorskip("torch")

from sgjm.research.cards import ExperimentCard, SweepResult
from sgjm.research.runner import _apply_override, _make_variant_config, run_sweep
from sgjm.research.sweep import (
    Sweep,
    SweepEntry,
    ablation_sweep,
    available_sweeps,
    get_sweep,
)
from sgjm.training.config import TrainingConfig


def test_sweep_registry_lists_known_sweeps():
    assert "ablation" in available_sweeps()
    assert "loss_weight" in available_sweeps()
    sweep = get_sweep("ablation")
    assert len(sweep) >= 3
    for entry in sweep:
        assert isinstance(entry.card, ExperimentCard)
        assert entry.card.hypothesis


def test_apply_override_nested_loss_weights():
    base = TrainingConfig.smoke()
    new_cfg, eval_overrides = _apply_override(base, "loss.jepa", 0.0)
    assert eval_overrides == {}
    assert new_cfg.loss.jepa == 0.0
    assert base.loss.jepa != 0.0  # original unchanged


def test_apply_override_model_block_size():
    base = TrainingConfig.smoke()
    new_cfg, _ = _apply_override(base, "model.block_size", 8)
    assert new_cfg.model.block_size == 8


def test_apply_override_eval_prefix_routes_to_eval_overrides():
    base = TrainingConfig.smoke()
    new_cfg, eval_overrides = _apply_override(base, "_eval.merge_radius_bits", 12)
    assert eval_overrides == {"merge_radius_bits": 12}
    assert new_cfg is base or new_cfg.model.block_size == base.model.block_size


def test_make_variant_config_applies_all_overrides(tmp_path):
    base = TrainingConfig.smoke()
    card = ExperimentCard(
        name="t",
        hypothesis="h",
        overrides={"loss.jepa": 0.0, "loss.drafter": 0.0, "_eval.merge_radius_bits": 2},
    )
    cfg, eo = _make_variant_config(base, card, tmp_path)
    assert cfg.loss.jepa == 0.0
    assert cfg.loss.drafter == 0.0
    assert cfg.checkpoint_dir.endswith("/t")
    assert eo["merge_radius_bits"] == 2


def test_sweep_result_primary_score_handles_error():
    card = ExperimentCard(name="x", hypothesis="h", overrides={})
    err_result = SweepResult(card=card, elapsed_sec=0, sgjm_metrics=None,
                              baseline_metrics=None, comparison=None, error="boom")
    assert err_result.primary_score == float("-inf")


def test_run_smoke_ablation_sweep_end_to_end(tmp_path):
    base_cfg = TrainingConfig.smoke()
    base_cfg.optim.max_steps = 2
    # Pick the first two entries to keep the test fast
    sweep = ablation_sweep()
    sweep.entries = sweep.entries[:2]
    results = run_sweep(sweep, base_cfg, backend="cpu", out_dir=tmp_path, eval_batches=2)
    assert len(results) == 2
    assert all(r.error is None for r in results)
    assert (tmp_path / "summary.json").exists()
    for r in results:
        assert r.sgjm_metrics is not None
        assert r.baseline_metrics is not None
        assert r.comparison is not None
