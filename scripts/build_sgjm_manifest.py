#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

EXCLUDE = {".git", "__pycache__", ".pytest_cache", "node_modules", "dist", "build", "venv", ".venv"}


def iter_files(root: Path, exts: set[str]):
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in EXCLUDE for part in p.parts):
            continue
        if p.suffix.lower() in exts:
            yield p


def conflict_score(text: str) -> float:
    keys = ["however", "but", "except", "unless", "fallback", "retry", "error", "TODO", "FIXME", "assert", "if", "else", "elif"]
    c = sum(text.lower().count(k) for k in keys)
    return min(1.0, c / 40.0)


def latent_horizon(text: str) -> int:
    # heuristic: longer structured code/docs = longer horizon
    lines = text.count("\n") + 1
    return 2 if lines < 40 else 4 if lines < 140 else 8


def verifier_hardness(text: str, cscore: float) -> str:
    if cscore > 0.65 or "except" in text.lower() or "edge case" in text.lower():
        return "hard"
    if cscore > 0.3:
        return "medium"
    return "easy"


def mergeability_bucket(text: str) -> str:
    uniq = len(set(re.findall(r"[A-Za-z_]{2,}", text)))
    if uniq < 120:
        return "high"
    if uniq < 300:
        return "medium"
    return "low"


def lane_for_path(p: Path) -> str:
    s = str(p).lower()
    if any(k in s for k in ["test", "spec", "benchmark", "bench"]):
        return "adversarial_branch_conflict"
    if p.suffix.lower() in {".md", ".rst", ".txt"}:
        return "reasoning_trajectories"
    return "code_long_context"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="data/sgjm_manifest.jsonl")
    ap.add_argument("--max-files", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    exts = {".py", ".md", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml"}
    files = list(iter_files(root, exts))
    rng.shuffle(files)
    files = files[: args.max_files]

    n = 0
    lane_counts = {"code_long_context": 0, "reasoning_trajectories": 0, "adversarial_branch_conflict": 0}
    with out.open("w") as f:
        for p in files:
            try:
                text = p.read_text(errors="ignore")
            except Exception:
                continue
            if len(text.strip()) < 40:
                continue
            cscore = conflict_score(text)
            rec = {
                "id": hashlib.sha256(str(p).encode()).hexdigest()[:16],
                "path": str(p),
                "lane": lane_for_path(p),
                "branch_conflict_score": round(cscore, 4),
                "latent_horizon": latent_horizon(text),
                "verifier_hardness": verifier_hardness(text, cscore),
                "mergeability_bucket": mergeability_bucket(text),
                "contradiction_tag": "local" if cscore > 0.35 else "none",
                "n_chars": len(text),
            }
            lane_counts[rec["lane"]] += 1
            f.write(json.dumps(rec) + "\n")
            n += 1

    summary = {
        "root": str(root),
        "out": str(out),
        "records": n,
        "lane_counts": lane_counts,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
