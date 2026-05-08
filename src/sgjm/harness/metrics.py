from __future__ import annotations

from dataclasses import dataclass, field

from sgjm.branch.lifecycle import StepReport


@dataclass(frozen=True)
class MetricSnapshot:
    steps: int
    drafted: int
    pruned: int
    accepted: int
    merged: int
    committed: int

    @property
    def acceptance_rate(self) -> float:
        return 0.0 if self.drafted == 0 else self.accepted / self.drafted

    @property
    def merge_rate(self) -> float:
        return 0.0 if self.accepted == 0 else self.merged / self.accepted

    @property
    def prune_rate(self) -> float:
        return 0.0 if self.drafted == 0 else self.pruned / self.drafted


@dataclass
class Metrics:
    steps: int = 0
    drafted: int = 0
    pruned: int = 0
    accepted: int = 0
    merged: int = 0
    committed: int = 0
    history: list[StepReport] = field(default_factory=list)

    def record(self, report: StepReport) -> None:
        self.steps += 1
        self.drafted += report.drafted
        self.pruned += report.pruned
        self.accepted += report.accepted
        self.merged += report.merged
        self.committed += len(report.committed_addresses)
        self.history.append(report)

    def snapshot(self) -> MetricSnapshot:
        return MetricSnapshot(
            steps=self.steps,
            drafted=self.drafted,
            pruned=self.pruned,
            accepted=self.accepted,
            merged=self.merged,
            committed=self.committed,
        )
