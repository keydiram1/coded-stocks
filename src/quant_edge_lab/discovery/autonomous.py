"""Autonomous preregistered funnel. Does not change frozen specs or start OOS/trading."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.console import Console

from quant_edge_lab.discovery.correct_decisions import stage2_eligible_keys
from quant_edge_lab.discovery.knowledge import append_jsonl, atomic_write_json, knowledge_dir
from quant_edge_lab.discovery.runner import load_state, run_batch
from quant_edge_lab.discovery.stages import StageName
from quant_edge_lab.statistics.summarize import benjamini_hochberg

console = Console()

STAGE1_SURVIVORS = [
    "H001.v1",
    "H004",
    "H004.v1",
    "H007",
    "H007.v1",
    "H028.v1",
    "H028.v2",
    "H037",
    "H037.v1",
    "H040",
    "H040.v1",
]

# Same frozen triggers, opposite side. Thresholds not fit to S1 magnitudes.
EXPLORATORY_JOBS: list[tuple[str, str, dict]] = [
    ("H002", "H002.fade", {"near": 0.003, "side": "short", "expected_direction": "short"}),
    ("H003", "H003.cont", {"ret_15m": 0.15, "ret_5m": 0.08, "side": "long", "expected_direction": "long"}),
]


def _passed_keys(state: dict[str, Any]) -> list[str]:
    out = []
    for k, j in (state.get("jobs") or {}).items():
        if j.get("state") == "ERROR":
            continue
        if j.get("decision") == "PASS":
            out.append(k)
    return sorted(out)


def _bh_jobs(state: dict[str, Any]) -> list[dict[str, Any]]:
    keys: list[str] = []
    ps: list[float] = []
    for k, j in sorted((state.get("jobs") or {}).items()):
        prim = ((j.get("stage_stats") or {}).get("primary") or {})
        p = prim.get("bootstrap_p")
        if p is None:
            p = prim.get("ttest_p")
        if p is None:
            continue
        keys.append(k)
        ps.append(float(p))
    rows = benjamini_hochberg(ps, q=0.05)
    for row, k in zip(rows, keys, strict=False):
        row["job"] = k
    return rows


def record_derived_hypotheses(root: Path) -> Path:
    """Lineage-only derived ideas. Do not splice into the parent batch."""
    recs = [
        {
            "id": "H002.fade",
            "parent": "H002",
            "origin": "derived_from_result",
            "relationship": "SAME frozen hold-near-high trigger, OPPOSITE side (short)",
            "why": "Stage-1 long continuation after hold-near-high had negative 15m mean; fade is the directional opposite, not a new threshold.",
            "queue": "exploratory",
            "not_in_parent_batch10": True,
            "thresholds_from_parent_frozen": True,
        },
        {
            "id": "H003.cont",
            "parent": "H003",
            "origin": "derived_from_result",
            "relationship": "SAME frozen parabolic trigger, OPPOSITE side (long continuation)",
            "why": "Stage-1 short exhaustion had large positive 15m mean (continuation, not fade). Opposite uses the same a-priori trigger.",
            "queue": "exploratory",
            "not_in_parent_batch10": True,
            "thresholds_from_parent_frozen": True,
        },
        {
            "id": "OBS-H028-fade-vs-continuation",
            "parent": "H028",
            "origin": "derived_from_result",
            "relationship": "Already registered H028.v1 fade vs primary continuation",
            "why": "Primary AH continuation killed on S1; fade variant passed. No new spec.",
            "queue": "parent_variants_already_running",
            "not_in_parent_batch10": False,
        },
    ]
    path = knowledge_dir(root) / "experiments" / "derived_from_batch10_s1.json"
    payload = {
        "type": "derived_hypotheses",
        "source_batch": "batch10-fast",
        "frozen_spec_untouched": "knowledge/families/batch10_frozen_v1.yaml",
        "created_at": datetime.now(UTC).isoformat(),
        "hypotheses": recs,
        "note": "SIGNAL_ONLY observations. Not substituted into parent experiment. BH: exploratory tests are a separate family.",
    }
    atomic_write_json(path, payload)
    append_jsonl(knowledge_dir(root) / "index.jsonl", {"type": "derived_hypotheses", "path": str(path)})
    return path


def run_stage(
    root: Path,
    *,
    batch_id: str,
    stage: StageName,
    job_keys: list[str] | None = None,
    extra_jobs: list[tuple[str, str, dict]] | None = None,
    print_every: int = 5,
) -> dict[str, Any]:
    console.print(f"\n=== FUNNEL {batch_id} stage={stage} jobs={job_keys or extra_jobs} ===")
    family_ids = ["H001"]  # placeholder overwritten by job_keys/extra_jobs
    return run_batch(
        root,
        family_ids,
        batch_id=batch_id,
        stage=stage,
        resume=True,
        job_keys=job_keys,
        extra_jobs=extra_jobs,
        compact_returns=True,
        print_every=print_every,
        expand_variants=False,
    )


def write_funnel_report(root: Path, funnel: dict[str, Any]) -> Path:
    reports = root / "reports" / "discovery"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "batch10_funnel_v1.md"
    lines = [
        "# Batch-10 research funnel (SIGNAL_ONLY)",
        "",
        "Not an edge. Not sealed OOS. Not trading. Frozen specs were not modified.",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Funnel",
        "",
    ]
    for stage_name, st in funnel.get("stages", {}).items():
        lines.append(f"### {stage_name}")
        lines.append("")
        lines.append(f"- batch_id: `{st.get('batch_id')}`")
        lines.append(f"- wall_s: {st.get('wall_s')}")
        lines.append(f"- passed: {st.get('passed')}")
        lines.append(f"- killed: {st.get('killed')}")
        lines.append(f"- errors: {st.get('errors')}")
        lines.append("")
        lines.append("| Job | Decision | Reason | Events | Ticker-days | Days | 15m mean | median | win | CI | top-ticker |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
        jobs = (st.get("state") or {}).get("jobs") or {}
        for k in sorted(jobs):
            j = jobs[k]
            prim = ((j.get("stage_stats") or {}).get("primary") or {})
            mean = prim.get("mean", (j.get("interim_means") or {}).get("15m"))
            med = prim.get("median")
            win = prim.get("win_rate")
            ci = prim.get("bootstrap_ci_95")
            ci_s = f"[{ci[0]:.4%}, {ci[1]:.4%}]" if isinstance(ci, list) and ci[0] is not None else "—"
            m_s = f"{mean:.4%}" if isinstance(mean, float) else "—"
            d_s = f"{med:.4%}" if isinstance(med, float) else "—"
            w_s = f"{win:.3f}" if isinstance(win, float) else "—"
            reason = (j.get("decision_reason") or "")[:80].replace("|", "/")
            lines.append(
                f"| {k} | {j.get('decision')} | {reason} | {j.get('events') or 0} | {j.get('ticker_days') or 0} | "
                f"{j.get('trading_days_with_events') or 0} | {m_s} | {d_s} | {w_s} | {ci_s} | "
                f"{j.get('concentration_top_ticker_share') or 0:.4f} |"
            )
        lines.append("")
        bh = st.get("bh") or []
        if bh:
            lines.append("BH on 15m bootstrap p (this stage's jobs only):")
            for row in bh:
                lines.append(f"- {row.get('job')}: p={row.get('p')} rejected={row.get('bh_rejected')}")
            lines.append("")
    lines += [
        "## Derived hypotheses (not substituted into parent)",
        "",
        "See `knowledge/experiments/derived_from_batch10_s1.json`. Exploratory queue uses identical frozen triggers with opposite side.",
        "",
        "## Warnings",
        "",
        "- SIGNAL_ONLY next-bar-open forwards; no spread/borrow/fill.",
        "- Variant-direction S1 audit: H007.v1 / H037.v1 / H040.v1 originally scored with parent short; corrected to frozen long without rescanning bars.",
        "- H004 primary metric is mean |15m|, not a tradable side.",
        "- Do not treat PASS as PROMISING or an edge.",
        "",
        "## Recommended next research (not started)",
        "",
        "- Sealed OOS only after a written freeze of survivors and BH family (requires approval).",
        "- Trade/quote validation only if a survivor still has a signed, clustered effect after full-panel robustness.",
        "- Do not paper/live trade from this report.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_autonomous_funnel(root: Path) -> dict[str, Any]:
    root = Path(root)
    t_all = datetime.now(UTC)
    record_derived_hypotheses(root)
    s1 = load_state(root, "batch10-fast")
    survivors = stage2_eligible_keys(s1) if s1.get("jobs") else list(STAGE1_SURVIVORS)
    survivors = [k for k in survivors if k in STAGE1_SURVIVORS] or list(STAGE1_SURVIVORS)

    funnel: dict[str, Any] = {
        "stage1_survivors": survivors,
        "stages": {},
        "derived": str(knowledge_dir(root) / "experiments" / "derived_from_batch10_s1.json"),
    }

    def snap(batch_id: str, stage: str, t0: datetime, state: dict) -> dict[str, Any]:
        jobs = state.get("jobs") or {}
        return {
            "batch_id": batch_id,
            "stage": stage,
            "wall_s": (datetime.now(UTC) - t0).total_seconds(),
            "passed": [k for k, j in jobs.items() if j.get("decision") == "PASS"],
            "killed": [k for k, j in jobs.items() if j.get("decision") == "KILL"],
            "errors": [k for k, j in jobs.items() if j.get("state") == "ERROR"],
            "bh": _bh_jobs(state),
            "state": {"jobs": {k: {kk: j.get(kk) for kk in (
                "decision", "decision_reason", "state", "events", "ticker_days",
                "trading_days_with_events", "interim_means", "stage_stats",
                "concentration_top_ticker_share", "robustness", "error",
            ) if kk in j} for k, j in jobs.items()}},
        }

    t0 = datetime.now(UTC)
    st2 = run_stage(root, batch_id="batch10-broad", stage="broad", job_keys=survivors, print_every=5)
    funnel["stages"]["stage2_broad"] = snap("batch10-broad", "broad", t0, st2)
    s2_pass = _passed_keys(st2)
    console.print(f"Stage 2 PASS: {s2_pass} KILL: {funnel['stages']['stage2_broad']['killed']}")

    if s2_pass:
        t0 = datetime.now(UTC)
        stf = run_stage(root, batch_id="batch10-full", stage="full", job_keys=s2_pass, print_every=10)
        funnel["stages"]["full_discovery"] = snap("batch10-full", "full", t0, stf)
        console.print(
            f"Full PASS: {funnel['stages']['full_discovery']['passed']} "
            f"KILL: {funnel['stages']['full_discovery']['killed']}"
        )
    else:
        console.print("No Stage-2 survivors; skipping full panel.")

    t0 = datetime.now(UTC)
    ste = run_stage(
        root,
        batch_id="batch10-explore-fast",
        stage="fast",
        extra_jobs=EXPLORATORY_JOBS,
        print_every=5,
    )
    funnel["stages"]["exploratory_s1"] = snap("batch10-explore-fast", "fast", t0, ste)
    exp_pass = _passed_keys(ste)
    if exp_pass:
        exp_jobs = [j for j in EXPLORATORY_JOBS if (j[1] if j[1] != "default" else j[0]) in exp_pass]
        t0 = datetime.now(UTC)
        ste2 = run_stage(
            root,
            batch_id="batch10-explore-broad",
            stage="broad",
            extra_jobs=exp_jobs,
            print_every=5,
        )
        funnel["stages"]["exploratory_s2"] = snap("batch10-explore-broad", "broad", t0, ste2)
        exp2_pass = _passed_keys(ste2)
        if exp2_pass:
            exp_jobs2 = [j for j in exp_jobs if (j[1] if j[1] != "default" else j[0]) in exp2_pass]
            t0 = datetime.now(UTC)
            stef = run_stage(
                root,
                batch_id="batch10-explore-full",
                stage="full",
                extra_jobs=exp_jobs2,
                print_every=10,
            )
            funnel["stages"]["exploratory_full"] = snap("batch10-explore-full", "full", t0, stef)

    funnel["wall_s_total"] = (datetime.now(UTC) - t_all).total_seconds()
    funnel["oos_started"] = False
    funnel["trading_started"] = False
    report = write_funnel_report(root, funnel)
    funnel["report"] = str(report)
    atomic_write_json(knowledge_dir(root) / "experiments" / "batch10_funnel_summary.json", funnel)
    console.print(f"Report: {report}")
    console.print("Funnel exhausted. No sealed OOS. No orders.")
    return funnel
