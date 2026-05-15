#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

KW = re.compile(
    r"(agent|speculative|reason|retriev|tool use|planning|verifier|benchmark|memory)",
    re.IGNORECASE,
)

@dataclass
class VariantMetrics:
    name: str
    params_m: float
    data_source: str
    corpus_bytes: int
    best_token_nll: float
    best_accept_rate: float
    notes: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_local_variants() -> list[VariantMetrics]:
    gate = _load_json(ROOT / "results/phase5-eval-gate/gate_report.json")
    c25 = _load_json(ROOT / "results/sgjm-25m-mlx-run1/config.json")
    c250 = _load_json(ROOT / "results/sgjm-250m-mlx-run1/config.json")

    v25 = VariantMetrics(
        name="sgjm-25m",
        params_m=25.0,
        data_source=c25["data_source"],
        corpus_bytes=c25["corpus_bytes"],
        best_token_nll=float(gate["sgjm"]["token_nll"]),
        best_accept_rate=float(gate["sgjm"]["branch_acceptance_rate"]),
        notes="Gate-pass config; tiny corpus, likely saturated; near-perfect acceptance.",
    )

    # 250M metrics from README-documented best step; stored in run readme.
    readme = (ROOT / "results/sgjm-250m-mlx-run1/README.md").read_text()
    nll = re.search(r"Best eval total loss: \*\*([0-9.]+)\*\*.*?\*\*([0-9.]+)\*\* \| \*\*99\.1%\*\*", readme, re.S)
    token_nll = 0.889
    if nll:
        token_nll = float(nll.group(2))

    v250 = VariantMetrics(
        name="sgjm-250m",
        params_m=251.0,
        data_source=c250["data_source"],
        corpus_bytes=c250["corpus_bytes"],
        best_token_nll=token_nll,
        best_accept_rate=0.991,
        notes="Converged then plateaued at 32 MiB python corpus capacity ceiling.",
    )
    return [v25, v250]


def fetch_arxiv_rss(category: str, limit: int = 12) -> list[dict]:
    url = f"https://rss.arxiv.org/rss/{category}"
    req = urllib.request.Request(url, headers={"User-Agent": "SGJM-AutoResearch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        xml = r.read()
    root = ET.fromstring(xml)
    out = []
    for item in root.findall("./channel/item"):
        title = item.findtext("title") or ""
        desc = item.findtext("description") or ""
        if not KW.search(title + " " + desc):
            continue
        link = item.findtext("link") or ""
        pub = item.findtext("pubDate") or ""
        dt = parsedate_to_datetime(pub) if pub else datetime.now(timezone.utc)
        out.append(
            {
                "category": category,
                "title": title.strip(),
                "link": link.strip(),
                "published": dt.isoformat(),
                "summary": re.sub(r"\s+", " ", desc).strip()[:420],
            }
        )
        if len(out) >= limit:
            break
    return out


def rank_papers(items: list[dict]) -> list[dict]:
    def score(x: dict) -> float:
        t = x["title"].lower()
        s = 0.0
        for k, w in [
            ("agent", 2.0),
            ("verifier", 2.0),
            ("reason", 1.5),
            ("retrieval", 1.3),
            ("memory", 1.2),
            ("benchmark", 1.1),
            ("speculative", 1.3),
        ]:
            if k in t:
                s += w
        # recency tie-break
        try:
            s += datetime.fromisoformat(x["published"]).timestamp() / 1e12
        except Exception:
            pass
        return s

    ranked = sorted(items, key=score, reverse=True)
    for r in ranked:
        r["score"] = round(score(r), 4)
    return ranked


def build_report(variants: list[VariantMetrics], papers: list[dict]) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines = []
    lines.append(f"# SGJM AutoResearch Report\n")
    lines.append(f"Generated: {now}\n")
    lines.append("## Variant Snapshot\n")
    for v in variants:
        lines.append(
            f"- {v.name}: params={v.params_m:.1f}M, data={v.data_source}, corpus={v.corpus_bytes/2**20:.1f} MiB, "
            f"best_token_nll={v.best_token_nll:.4f}, accept={v.best_accept_rate:.3f}. {v.notes}"
        )

    lines.append("\n## Diagnosis (first principles)\n")
    lines.append("1) 25M is architecture-validated but data-understressed (tiny corpus, near-perfect acceptance).")
    lines.append("2) 250M is data-bottlenecked (32 MiB corpus ceiling), not capacity-limited.")
    lines.append("3) Merge precision and verifier utility are the real SGJM differentiators; dataset must stress branch disagreement, not just next-token CE.")

    lines.append("\n## Latest arXiv Signals (keyword-filtered)\n")
    for p in papers[:12]:
        lines.append(f"- [{p['title']}]({p['link']}) | {p['category']} | score={p['score']}")

    lines.append("\n## Optimization A: Dataset Design (capability-targeted)\n")
    lines.append("- Build a three-lane mixture with fixed weights: 40% code long-context, 35% reasoning trajectories, 25% adversarial branch-conflict samples.")
    lines.append("- Add SGJM-specific labels per sample: branch conflict score, latent transition smoothness, verifier hardness, mergeability bucket.")
    lines.append("- Curriculum by block length: start with block=2 tasks, then 30% block=4, finally 10% block=8 hard cases.")
    lines.append("- Data scale targets: 25M=8-12B tokens, 250M=35-60B, 1B=120-220B. Current corpora are orders of magnitude too small.")

    lines.append("\n## Optimization B: Training Method (25M/250M/1B)\n")
    lines.append("- Two-stage schedule: Stage-1 token+JEPA warm start, Stage-2 full SGJM with verifier anneal.")
    lines.append("- Dynamic loss weighting: jepa 0.05->0.25 ramp, verifier 0.0->0.1 ramp; keep drafter 0.5 until accept>0.6 then decay to 0.35.")
    lines.append("- Acceptance-controlled LR: if accept<0.45 for 3 evals, reduce LR 20% and increase verifier margin mining.")
    lines.append("- Model-size specifics: 25M prioritize robustness/regularization; 250M prioritize throughput+long context; 1B use FSDP/ZeRO + GQA + RoPE scaling and staged context extension.")

    lines.append("\n## Concrete Retrain Targets\n")
    lines.append("- 25M: token_nll <= baseline+0.03, accept>=0.70, merge_adv>=2.0, compute_adv>=4.0")
    lines.append("- 250M: token_nll <= baseline+0.025, accept>=0.75, merge_adv>=2.5, compute_adv>=6.0")
    lines.append("- 1B: token_nll <= baseline+0.02, accept>=0.80, merge_adv>=3.0, compute_adv>=8.0")

    return "\n".join(lines) + "\n"


def main() -> None:
    variants = load_local_variants()
    items = []
    for cat in ("cs.AI", "cs.CL", "cs.LG"):
        items.extend(fetch_arxiv_rss(cat, limit=10))
    ranked = rank_papers(items)

    out_dir = ROOT / "results" / "autoresearch"
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"papers_{ts}.json").write_text(json.dumps(ranked[:30], indent=2))
    report = build_report(variants, ranked)
    report_path = out_dir / f"sgjm_autoresearch_{ts}.md"
    report_path.write_text(report)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(report_path),
        "papers": str(out_dir / f"papers_{ts}.json"),
        "variants": [asdict(v) for v in variants],
    }
    (out_dir / "latest_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
