from __future__ import annotations

from dataclasses import dataclass, field

from sgjm.research.cards import ExperimentCard


@dataclass
class SweepEntry:
    card: ExperimentCard


@dataclass
class Sweep:
    name: str
    entries: list[SweepEntry] = field(default_factory=list)

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def ablation_sweep() -> Sweep:
    """Standard SGJM ablations: drop each auxiliary loss head, one at a time."""
    return Sweep(
        name="ablation",
        entries=[
            SweepEntry(ExperimentCard(
                name="sgjm_full",
                hypothesis="All four losses (token + drafter + jepa + verifier) contribute.",
                overrides={},
                expected_signal="Best on combined score; sets the ceiling.",
            )),
            SweepEntry(ExperimentCard(
                name="sgjm_no_jepa",
                hypothesis="JEPA pruning is unnecessary; verifier alone is enough.",
                overrides={"loss.jepa": 0.0},
                expected_signal="jepa_top1_acc drops to chance; branch_acceptance flat.",
            )),
            SweepEntry(ExperimentCard(
                name="sgjm_no_drafter",
                hypothesis="Drafter loss isn't needed; backbone hidden states are enough.",
                overrides={"loss.drafter": 0.0},
                expected_signal="Drafter outputs become incoherent; branch_acceptance drops.",
            )),
            SweepEntry(ExperimentCard(
                name="sgjm_no_verifier",
                hypothesis="Verifier loss isn't needed; rely on judge for acceptance.",
                overrides={"loss.verifier": 0.0},
                expected_signal="branch_acceptance_rate uninformative (~0.5).",
            )),
            SweepEntry(ExperimentCard(
                name="sgjm_token_only",
                hypothesis="Aux losses don't help; equivalent to baseline plus dead weight.",
                overrides={"loss.drafter": 0.0, "loss.jepa": 0.0, "loss.verifier": 0.0},
                expected_signal="Should approximate baseline NLL with all aux metrics dead.",
            )),
        ],
    )


def loss_weight_sweep() -> Sweep:
    """Vary the JEPA loss weight to find the elbow."""
    return Sweep(
        name="loss_weight",
        entries=[
            SweepEntry(ExperimentCard(
                name=f"jepa_w_{w}",
                hypothesis=f"jepa weight {w} balances aux signal vs token CE.",
                overrides={"loss.jepa": w},
                expected_signal="Find the smallest weight that keeps jepa_top1_acc above chance.",
            ))
            for w in (0.0, 0.05, 0.25, 1.0, 4.0)
        ],
    )


def block_size_sweep() -> Sweep:
    """Vary the drafter block size."""
    return Sweep(
        name="block_size",
        entries=[
            SweepEntry(ExperimentCard(
                name=f"block_{b}",
                hypothesis=f"block_size={b} hits the compute/acceptance tradeoff sweet spot.",
                overrides={"model.block_size": b},
                expected_signal="Larger blocks = higher compute/accepted, lower acceptance.",
            ))
            for b in (2, 4, 8)
        ],
    )


def merge_radius_sweep() -> Sweep:
    """Vary the SimHash merge radius used at eval time."""
    return Sweep(
        name="merge_radius",
        entries=[
            SweepEntry(ExperimentCard(
                name=f"merge_r{r}",
                hypothesis=f"merge_radius={r} bits trades recall vs precision.",
                overrides={"_eval.merge_radius_bits": r},
                expected_signal="Larger radius merges more aggressively (higher recall, lower precision).",
            ))
            for r in (2, 4, 6, 8, 12)
        ],
    )


_REGISTRY = {
    "ablation": ablation_sweep,
    "loss_weight": loss_weight_sweep,
    "block_size": block_size_sweep,
    "merge_radius": merge_radius_sweep,
}


def get_sweep(name: str) -> Sweep:
    if name not in _REGISTRY:
        raise KeyError(f"unknown sweep {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available_sweeps() -> list[str]:
    return sorted(_REGISTRY)
