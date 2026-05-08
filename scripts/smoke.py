from __future__ import annotations

from sgjm.harness.runner import HarnessConfig, HarnessRunner
from sgjm.modules.backbone import StubBackbone
from sgjm.modules.drafter import StubDrafter
from sgjm.modules.judge import StubJudge


def main() -> None:
    backbone = StubBackbone(latent_dim=16, seed=7)
    drafter = StubDrafter(backbone=backbone, vocab_size=32, seed=11)
    judge = StubJudge()
    runner = HarnessRunner(
        backbone=backbone,
        drafter=drafter,
        judge=judge,
        config=HarnessConfig(branches_per_step=4, block_size=3, max_steps=4, keep_top_k=2),
    )
    snap = runner.run(prompt_tokens=[1, 2, 3, 4])
    print(
        f"steps={snap.steps} drafted={snap.drafted} pruned={snap.pruned} "
        f"accepted={snap.accepted} merged={snap.merged} committed={snap.committed} "
        f"acc_rate={snap.acceptance_rate:.3f} merge_rate={snap.merge_rate:.3f} "
        f"prune_rate={snap.prune_rate:.3f} graph_size={len(runner.graph)}"
    )


if __name__ == "__main__":
    main()
