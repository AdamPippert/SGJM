from sgjm.harness.runner import HarnessConfig, HarnessRunner
from sgjm.modules.backbone import StubBackbone
from sgjm.modules.drafter import StubDrafter
from sgjm.modules.judge import StubJudge


def test_smoke_run_records_metrics():
    backbone = StubBackbone(latent_dim=16, seed=3)
    drafter = StubDrafter(backbone=backbone, vocab_size=16, seed=5)
    judge = StubJudge()
    runner = HarnessRunner(
        backbone=backbone,
        drafter=drafter,
        judge=judge,
        config=HarnessConfig(branches_per_step=3, block_size=2, max_steps=3, keep_top_k=2),
    )
    snap = runner.run(prompt_tokens=[1, 2, 3])
    assert snap.steps > 0
    assert snap.drafted >= snap.accepted
    assert snap.committed >= 1
    assert 0.0 <= snap.acceptance_rate <= 1.0
    assert len(runner.graph) >= 2
