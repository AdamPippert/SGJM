from sgjm.eval.checkpoint import load_checkpoint
from sgjm.eval.report import (
    BaselineEvalMetrics,
    ComparisonReport,
    SGJMEvalMetrics,
    compare,
)

__all__ = [
    "BaselineEvalMetrics",
    "ComparisonReport",
    "SGJMEvalMetrics",
    "compare",
    "evaluate_baseline",
    "evaluate_sgjm",
    "load_checkpoint",
]


def __getattr__(name: str):
    # evaluate_baseline/evaluate_sgjm live in sgjm.eval.metrics, which
    # imports torch. Deferred so `import sgjm.eval.<anything>` — which
    # always runs this __init__ first — never requires torch on an
    # MLX-only install unless one of these two names is actually used.
    if name in ("evaluate_baseline", "evaluate_sgjm"):
        from sgjm.eval import metrics

        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
