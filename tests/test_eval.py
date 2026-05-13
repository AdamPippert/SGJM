import pytest

torch = pytest.importorskip("torch")

from sgjm.eval.checkpoint import load_checkpoint
from sgjm.eval.metrics import (
    BaselineEvalMetrics,
    SGJMEvalMetrics,
    compare,
    evaluate_baseline,
    evaluate_sgjm,
)
from sgjm.training.config import TrainingConfig
from sgjm.training.data import ByteDataset, synthetic_corpus
from sgjm.training.torch_backend.baseline import BaselineLM
from sgjm.training.torch_backend.model import SGJM
from sgjm.training.torch_backend.trainer import train


def _smoke_cfg(arch: str, ckpt_dir) -> TrainingConfig:
    cfg = TrainingConfig.smoke()
    cfg.arch = arch
    cfg.checkpoint_dir = str(ckpt_dir)
    return cfg


def test_baseline_model_param_count_matches_sgjm_total():
    cfg = TrainingConfig.sgjm_25m()
    sgjm = SGJM(cfg.model)
    baseline = BaselineLM(cfg.model)
    sgjm_n = sgjm.num_parameters()
    base_n = baseline.num_parameters()
    # Should land within +/- 10% so the comparison is fair.
    assert 0.9 * sgjm_n <= base_n <= 1.1 * sgjm_n, (
        f"baseline {base_n/1e6:.2f}M vs sgjm {sgjm_n/1e6:.2f}M too different"
    )


def test_evaluate_sgjm_returns_finite_metrics():
    cfg = TrainingConfig.smoke()
    model = SGJM(cfg.model)
    corpus = synthetic_corpus(4096, seed=0)
    ds = ByteDataset(corpus, cfg.optim.seq_len)
    metrics = evaluate_sgjm(
        model,
        cfg,
        ds,
        n_batches=2,
        n_distractors=4,
        n_merge_pairs=128,
        drafts_per_step=2,
        device="cpu",
    )
    assert 0.0 <= metrics.branch_acceptance_rate <= 1.0
    assert 0.0 <= metrics.jepa_top1_acc <= 1.0
    assert metrics.compute_per_accepted_token > 0
    # merge_precision_js can legitimately be NaN when too few pairs land
    # inside the merge radius; everything else must be finite.
    nan_ok = {"merge_precision_js", "random_pair_js"}
    for k, v in metrics.to_dict().items():
        if isinstance(v, float) and k not in nan_ok:
            assert v == v, f"{k} is NaN"


def test_evaluate_baseline_returns_finite_metrics():
    cfg = TrainingConfig.smoke()
    model = BaselineLM(cfg.model)
    corpus = synthetic_corpus(4096, seed=0)
    ds = ByteDataset(corpus, cfg.optim.seq_len)
    metrics = evaluate_baseline(model, cfg, ds, n_batches=2, device="cpu")
    assert metrics.token_nll > 0
    assert metrics.compute_per_token > 0


def test_compare_constructs_report():
    sgjm = SGJMEvalMetrics(
        n_tokens=100, n_positions=50, token_nll=1.5, token_ppl=4.5,
        branch_acceptance_rate=0.6, jepa_top1_acc=0.4, jepa_chance_top1=0.1,
        merge_precision_js=0.01, random_pair_js=0.05,
        merge_precision_advantage=5.0,
        compute_per_accepted_token=1e6,
    )
    baseline = BaselineEvalMetrics(
        n_tokens=100, token_nll=1.5, token_ppl=4.5, compute_per_token=2e6,
    )
    report = compare(sgjm, baseline)
    assert report.gate_passed
    assert report.compute_advantage > 1.0


def test_compare_fails_gate_on_high_nll():
    sgjm = SGJMEvalMetrics(
        n_tokens=100, n_positions=50, token_nll=3.0, token_ppl=20.0,
        branch_acceptance_rate=0.6, jepa_top1_acc=0.4, jepa_chance_top1=0.1,
        merge_precision_js=0.01, random_pair_js=0.05,
        merge_precision_advantage=5.0,
        compute_per_accepted_token=1e6,
    )
    baseline = BaselineEvalMetrics(
        n_tokens=100, token_nll=1.5, token_ppl=4.5, compute_per_token=2e6,
    )
    report = compare(sgjm, baseline)
    assert not report.gate_passed
    assert any("nll" in r for r in report.gate_reasons)


def test_end_to_end_train_then_eval(tmp_path):
    sgjm_dir = tmp_path / "sgjm"
    base_dir = tmp_path / "baseline"
    sgjm_cfg = _smoke_cfg("sgjm", sgjm_dir)
    base_cfg = _smoke_cfg("baseline", base_dir)

    sgjm_result = train(sgjm_cfg, backend="cpu")
    base_result = train(base_cfg, backend="cpu")
    assert sgjm_result.checkpoint_path is not None
    assert base_result.checkpoint_path is not None

    sgjm_loaded = load_checkpoint(sgjm_result.checkpoint_path, device="cpu")
    base_loaded = load_checkpoint(base_result.checkpoint_path, device="cpu")
    assert sgjm_loaded.arch == "sgjm"
    assert base_loaded.arch == "baseline"

    corpus = synthetic_corpus(2048, seed=0)
    ds = ByteDataset(corpus, sgjm_cfg.optim.seq_len)
    sgjm_metrics = evaluate_sgjm(
        sgjm_loaded.model, sgjm_cfg, ds, n_batches=2,
        n_distractors=4, n_merge_pairs=64, drafts_per_step=2, device="cpu",
    )
    base_metrics = evaluate_baseline(
        base_loaded.model, base_cfg, ds, n_batches=2, device="cpu",
    )
    report = compare(sgjm_metrics, base_metrics)
    assert isinstance(report.gate_passed, bool)
    # Untrained smoke models won't pass the gate, just verify the report shape
    assert report.sgjm.token_nll > 0
    assert report.baseline.token_nll > 0
