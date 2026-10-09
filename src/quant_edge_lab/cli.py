from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from quant_edge_lab.data.sample import write_sample_data
from quant_edge_lab.hypotheses.loader import load_hypothesis
from quant_edge_lab.pipeline import reproduce_experiment, run_hypothesis
from quant_edge_lab.registry.store import list_experiments, load_experiment

app = typer.Typer(no_args_is_help=True, help="Quant Edge Lab — scientific event-study engine.")
massive_app = typer.Typer(no_args_is_help=True, help="Massive historical data ingest.")
app.add_typer(massive_app, name="massive")
hyp_app = typer.Typer(no_args_is_help=True, help="Hypothesis family registry.")
disc_app = typer.Typer(no_args_is_help=True, help="Staged discovery funnel.")
know_app = typer.Typer(no_args_is_help=True, help="Research knowledge base.")
app.add_typer(hyp_app, name="hypotheses")
app.add_typer(disc_app, name="discovery")
app.add_typer(know_app, name="knowledge")
console = Console()


@app.command("init-sample-data")
def init_sample_data(seed: int = 42) -> None:
    """Write deterministic synthetic bars (engine tests, not evidence)."""
    written = write_sample_data(Path.cwd(), seed=seed)
    console.print("Wrote sample data:")
    for k, v in written.items():
        console.print(f"  {k}: {v}")


@app.command("validate-hypothesis")
def validate_hypothesis(path: Path) -> None:
    spec = load_hypothesis(path)
    console.print(f"OK  {spec.id}  family={spec.family}  side={spec.side}")
    console.print(f"    conditions={len(spec.trigger.conditions)} features={list(spec.features)}")


@app.command("run")
def run(path: Path, dataset: str = "sample") -> None:
    if dataset in {"massive-flat", "massive-full", "massive-flatfiles"}:
        from quant_edge_lab.data.massive.flatfiles import DATASET
        from quant_edge_lab.pipeline_flat import run_gap_rvol_on_flat

        rec = run_gap_rvol_on_flat(path, root=Path.cwd(), dataset=DATASET)
    else:
        rec = run_hypothesis(path, root=Path.cwd(), dataset=dataset)
    console.print(f"experiment_id: {rec.experiment_id}")
    console.print(f"dataset:       {rec.dataset} ({rec.data_source})")
    console.print(f"events:        {rec.events_path}")
    console.print(f"outcomes:      {rec.outcomes_path}")
    console.print(f"report:        {rec.report_path}")
    console.print(f"stage:         {rec.stage}  (SIGNAL_ONLY)")


@app.command("list-experiments")
def list_exps() -> None:
    recs = list_experiments(Path.cwd())
    table = Table(title="Experiments")
    table.add_column("id")
    table.add_column("hypothesis")
    table.add_column("dataset")
    table.add_column("stage")
    for r in recs:
        table.add_row(r.experiment_id, r.hypothesis_id, r.dataset, r.stage)
    console.print(table)
    if not recs:
        console.print("No experiments yet.")


@app.command("show-experiment")
def show_experiment(experiment_id: str) -> None:
    rec = load_experiment(experiment_id, Path.cwd())
    console.print_json(rec.model_dump_json())
    summary = Path(rec.summary_path)
    if summary.exists():
        console.print(summary.read_text(encoding="utf-8"))


@app.command("reproduce")
def reproduce(experiment_id: str) -> None:
    result = reproduce_experiment(experiment_id, root=Path.cwd())
    console.print_json(data=result)
    if not result["match"]:
        raise typer.Exit(code=1)


@massive_app.command("check-access")
def massive_check_access() -> None:
    from quant_edge_lab.data.massive.access import check_access

    report = check_access(Path.cwd())
    console.print_json(data=report)


@massive_app.command("download")
def massive_download(
    dataset: str = typer.Option("massive-smoke", help="massive-smoke | massive-pilot | massive-full"),
    start: str | None = None,
    end: str | None = None,
    tickers: str | None = None,
    max_tickers: int | None = None,
) -> None:
    from quant_edge_lab.config import Paths
    from quant_edge_lab.data.massive.client import MassiveClient, probe_bases
    from quant_edge_lab.data.massive.ingest import ingest_full, ingest_pilot, ingest_smoke, ingest_tickers
    from quant_edge_lab.secrets import massive_api_key

    root = Path.cwd()
    key = massive_api_key(root)
    base = probe_bases(key)
    client = MassiveClient(api_key=key, base_url=base)
    if dataset == "massive-smoke" and not start:
        ck = ingest_smoke(client, root)
    elif tickers:
        if not start or not end:
            raise typer.BadParameter("--start and --end are required with --tickers")
        ck = ingest_tickers(
            client,
            root,
            dataset,
            [t.strip().upper() for t in tickers.split(",") if t.strip()],
            date.fromisoformat(start),
            date.fromisoformat(end),
        )
    elif dataset == "massive-pilot":
        s = date.fromisoformat(start) if start else date(2026, 9, 2)
        e = date.fromisoformat(end) if end else date(2026, 9, 30)
        ck = ingest_pilot(client, root, s, e, max_tickers=max_tickers or 40)
    elif dataset == "massive-full":
        s = date.fromisoformat(start) if start else date(2024, 10, 1)
        e = date.fromisoformat(end) if end else date(2026, 10, 1)
        ck = ingest_full(client, root, s, e, max_tickers=max_tickers)
    else:
        raise typer.BadParameter(f"Unknown dataset {dataset}")
    console.print(f"data_root: {Paths(root).data}")
    console.print_json(data={"dataset": dataset, "meta": ck.get("meta"), "failures": ck.get("failures", [])[:20], "data_root": str(Paths(root).data)})


@massive_app.command("status")
def massive_status(dataset: str = "massive-smoke") -> None:
    from quant_edge_lab.data.massive.ingest import status as rest_status
    from quant_edge_lab.data.massive.flatfiles import DATASET, flatfiles_status, load_manifest

    if dataset in {DATASET, "massive-flatfiles"}:
        console.print_json(data=flatfiles_status(Path.cwd()))
        return
    if dataset == "massive-full":
        console.print_json(
            data={
                "dataset": "massive-full",
                "note": "REST ticker ingest is leftover and not resumed. Flat Files is the active 1m history.",
                "rest_leftover": rest_status(Path.cwd(), "massive-full"),
                "flatfiles": flatfiles_status(Path.cwd()),
            }
        )
        return
    console.print_json(data=rest_status(Path.cwd(), dataset))


@massive_app.command("access-boundary")
def massive_access_boundary() -> None:
    from quant_edge_lab.data.massive.flatfiles import run_access_boundary

    report = run_access_boundary(Path.cwd())
    console.print_json(data=report)


@massive_app.command("prepare-reference")
def massive_prepare_reference() -> None:
    from quant_edge_lab.data.massive.reference import prepare_pit_reference

    console.print_json(data=prepare_pit_reference(Path.cwd()))


@massive_app.command("validate-flat")
def massive_validate_flat() -> None:
    from quant_edge_lab.data.massive.quality import certify_massive_flat

    console.print_json(data=certify_massive_flat(Path.cwd()))


@massive_app.command("certify-flat")
def massive_certify_flat() -> None:
    from quant_edge_lab.data.massive.quality import certify_massive_flat

    report = certify_massive_flat(Path.cwd())
    console.print(f"overall: {report.get('overall')}")
    console.print(f"report:  {report.get('report_md')}")
    console.print_json(
        data={k: report[k] for k in ("overall", "categories", "report_md", "report_json") if k in report}
    )


@massive_app.command("sync-flatfiles")
def massive_sync_flatfiles(
    start: str | None = None,
    end: str | None = None,
    workers: int = 6,
    convert: bool = True,
) -> None:
    from quant_edge_lab.data.massive.flatfiles import run_access_boundary, sync_range

    root = Path.cwd()
    if not start or not end:
        bound = run_access_boundary(root)
        gb = bound.get("get_boundary") or {}
        start = start or gb.get("earliest_get_ok")
        end = end or gb.get("latest_get_ok")
        if not start or not end:
            raise typer.Exit(code=1)
        if not bound.get("looks_like_starter_5y"):
            console.print("Access window is not ~5 years; refusing full catalog sync.")
            console.print_json(data=bound)
            raise typer.Exit(code=2)
    meta = sync_range(root, start, end, workers=workers, convert=convert)
    console.print_json(data=meta)


@hyp_app.command("list")
def hypotheses_list() -> None:
    from quant_edge_lab.discovery.catalog import all_families, write_registry

    write_registry(Path.cwd())
    table = Table(title="Registered families")
    table.add_column("id")
    table.add_column("title")
    table.add_column("exec")
    table.add_column("dir")
    table.add_column("engine")
    for f in all_families():
        table.add_row(f.id, f.title[:40], "yes" if f.executable else "idea", f.expected_direction, f.engine)
    console.print(table)


@hyp_app.command("show")
def hypotheses_show(family_id: str) -> None:
    from quant_edge_lab.discovery.catalog import family_by_id

    console.print_json(family_by_id(family_id).model_dump_json())


@disc_app.command("run")
def discovery_run(family_id: str, stage: str = "fast") -> None:
    from quant_edge_lab.discovery.runner import run_batch

    batch_id = f"single-{family_id}-{stage}"
    state = run_batch(Path.cwd(), [family_id], batch_id=batch_id, stage=stage)  # type: ignore[arg-type]
    console.print_json(data={k: state["jobs"][family_id] for k in state["jobs"]})


@disc_app.command("run-batch")
def discovery_run_batch(name: str = "smoke-executable", stage: str = "fast") -> None:
    from quant_edge_lab.discovery.catalog import all_families
    from quant_edge_lab.discovery.eval_batch10 import BATCH10_PRIMARY
    from quant_edge_lab.discovery.runner import run_batch

    if name == "initial-40":
        console.print("Refusing to launch all 40. Most are IDEA_ONLY. Use smoke-executable.")
        raise typer.Exit(code=2)
    if name == "batch10-fast":
        console.print("Refusing auto-launch. Use: python -m quant_edge_lab discovery batch10-fast")
        console.print("That command prints readiness. Add --launch only after you confirm.")
        raise typer.Exit(code=2)
    ids = [f.id for f in all_families() if f.executable]
    if name == "smoke-executable":
        ids = [i for i in ids if i in {"H001", "H040"}]
    elif name.startswith("batch10"):
        ids = list(BATCH10_PRIMARY)
    state = run_batch(Path.cwd(), ids, batch_id=name, stage=stage)  # type: ignore[arg-type]
    console.print(f"batch {name} finished. state -> data/derived/discovery/batches/{name}/state.json")
    console.print("This is a screen, SIGNAL_ONLY, INTERIM until full discovery. Not an edge.")


@disc_app.command("batch10-fast")
def discovery_batch10_fast(
    launch: bool = typer.Option(
        False,
        "--launch",
        help="Run the 50-day market FAST SCREEN. Default is readiness only (does not touch the panel).",
    ),
) -> None:
    from quant_edge_lab.data.massive.flatfiles import load_manifest, local_parquet_path
    from quant_edge_lab.discovery.eval_batch10 import BATCH10_PRIMARY, BATCH10_VARIANTS, expand_batch10_jobs
    from quant_edge_lab.discovery.runner import run_batch
    from quant_edge_lab.discovery.stages import select_stage_days

    root = Path.cwd()
    jobs = expand_batch10_jobs(list(BATCH10_PRIMARY), expand_variants=True)
    n_var = sum(len(v) for v in BATCH10_VARIANTS.values())
    console.print("FROZEN spec: knowledge/families/batch10_frozen_v1.yaml")
    console.print(f"Families: {', '.join(BATCH10_PRIMARY)}")
    console.print(f"Jobs: {len(jobs)} ({len(BATCH10_PRIMARY)} primary + {n_var} neighborhood variants)")
    console.print("Stage-1 outcomes: vectorized forward close-join; MFE/MAE/barriers skipped")
    console.print("Shared scan: one parquet+TZ per day for all jobs")
    n_panel = 50
    try:
        man = load_manifest(root)
        all_days = sorted(
            d
            for d, rec in man.get("files", {}).items()
            if rec.get("conversion_status") == "ok" and local_parquet_path(root, d).exists()
        )
        n_panel = len(select_stage_days(all_days, "fast"))
        console.print(f"FAST panel: {n_panel} hashed days (stage_panels_v1) of {len(all_days)} converted")
    except Exception as exc:
        console.print(f"FAST panel: 50 days (manifest not read: {type(exc).__name__})")
    console.print("Expected runtime: ~5–12 min wall (shared IO ~2s/day + vectorized S1). Worst ~20 min if a day is huge.")
    console.print("SIGNAL_ONLY. PASS is more compute, not an edge. Do not tune from this screen.")
    cmd = "python -m quant_edge_lab discovery batch10-fast --launch"
    console.print(f"Launch command (not executed unless --launch): {cmd}")
    if not launch:
        console.print("Readiness only. Market batch not started.")
        raise typer.Exit(code=0)
    state = run_batch(
        root,
        list(BATCH10_PRIMARY),
        batch_id="batch10-fast",
        stage="fast",
        expand_variants=True,
    )
    console.print("batch10-fast finished. SIGNAL_ONLY INTERIM. Not an edge.")
    console.print(f"jobs={len(state.get('jobs') or {})}")


@disc_app.command("auto-funnel")
def discovery_auto_funnel() -> None:
    """Run Stage 2 → full discovery → exploratory queue. No OOS/trading."""
    from quant_edge_lab.discovery.autonomous import run_autonomous_funnel

    funnel = run_autonomous_funnel(Path.cwd())
    console.print(f"report={funnel.get('report')}")
    console.print(f"wall_s={funnel.get('wall_s_total')}")
    console.print("SIGNAL_ONLY. Funnel exhausted. No sealed OOS. No orders.")


@disc_app.command("two-workers")
def discovery_two_workers(
    launch: bool = typer.Option(False, "--launch", help="Start Worker A + Worker B (shared scan)."),
) -> None:
    """Directional H004 (A) + next-gen discovery (B). Frozen dir_gates_v1. No OOS/trading."""
    from quant_edge_lab.discovery.dir_gates import freeze_hashes, load_gates
    from quant_edge_lab.discovery.two_workers import run_two_workers

    root = Path.cwd()
    gates = load_gates(root)
    hashes = freeze_hashes(root)
    console.print(f"Frozen gates {gates.version} hash={hashes['gates'][:16]}…")
    console.print(f"Frozen batch hash={hashes['batch'][:16]}…")
    console.print("Logical workers A/B share one daily parquet scan. SIGNAL_ONLY.")
    if not launch:
        console.print("Readiness only. Pass --launch to start.")
        raise typer.Exit(code=0)
    state = run_campaign_v2(root)
    console.print(f"report={state.get('report')}")
    console.print("SIGNAL_ONLY. Funnel exhausted. No sealed OOS. No orders.")


@disc_app.command("campaign-v2")
def discovery_campaign_v2(
    launch: bool = typer.Option(False, "--launch", help="Start directional campaign v2 (shared causal store)."),
) -> None:
    """Frozen A–L directional campaign. No OOS/trading."""
    from quant_edge_lab.discovery.campaign_v2_runner import freeze_campaign, run_campaign_v2

    root = Path.cwd()
    fr = freeze_campaign(root)
    console.print(f"Frozen v2 manifest={fr['manifest'][:16]}… gates={fr['gates'][:16]}… git={fr['git']}")
    console.print("Shared causal feature store. SIGNAL_ONLY.")
    if not launch:
        console.print("Readiness only. Pass --launch to start.")
        raise typer.Exit(code=0)
    state = run_campaign_v2(root)
    console.print(f"report={state.get('report')}")
    console.print("SIGNAL_ONLY. Campaign exhausted. No sealed OOS. No orders.")


@disc_app.command("campaign-v3")
def discovery_campaign_v3(
    launch: bool = typer.Option(False, "--launch", help="Start directional campaign v3 (news+XS)."),
    smoke: bool = typer.Option(False, "--smoke", help="Two hashed news-period days, representative news jobs only."),
    resume_unevaluated: bool = typer.Option(False, "--resume-unevaluated", help="Resume 53 news jobs; keep XR results."),
) -> None:
    """Frozen news+cross-section campaign. No OOS/trading. Does not modify V2."""
    from quant_edge_lab.discovery.campaign_v3_eval import load_v3_jobs
    from quant_edge_lab.discovery.campaign_v3_runner import (
        SMOKE_NEWS_JOB_IDS,
        freeze_campaign,
        run_campaign_v3,
    )

    root = Path.cwd()
    fr = freeze_campaign(root)
    _, jobs = load_v3_jobs(root)
    console.print(f"Frozen v3 n_jobs={len(jobs)} manifest={fr['manifest'][:16]}… gates={fr['gates'][:16]}… git={fr['git']}")
    console.print("PIT: news_available_at=published_utc+60m. sentiment_reasoning audit-only. SIGNAL_ONLY.")
    if not launch and not smoke:
        console.print("Readiness only. Pass --launch to ingest news (if needed) and run.")
        raise typer.Exit(code=0)
    if smoke:
        state = run_campaign_v3(
            root,
            smoke_days=2,
            job_ids=list(SMOKE_NEWS_JOB_IDS),
            batch_id="dir-campaign-v3-news-join-smoke",
        )
    else:
        state = run_campaign_v3(root, resume_unevaluated=resume_unevaluated or True)
    if state.get("stop"):
        console.print(f"STOPPED: {state.get('reason') or state.get('news', {}).get('reason')}")
        raise typer.Exit(code=1)
    console.print(f"report={state.get('report')}")
    console.print("SIGNAL_ONLY. Campaign exhausted. No sealed OOS. No orders.")


@disc_app.command("campaign-v4r")
def discovery_campaign_v4r(
    execute: bool = typer.Option(False, "--execute", help="Full V4R after tests/preflight GO."),
    smoke: bool = typer.Option(False, "--smoke", help="Tiny synthetic/real slice only."),
) -> None:
    """V4R repaired discovery. Does not overwrite V4. No sealed OOS."""
    from quant_edge_lab.discovery.v4r.runner import freeze_v4r, load_v4r, run_campaign_v4r

    root = Path.cwd()
    fr = freeze_v4r(root)
    man, gates = load_v4r(root)
    console.print(f"FROZEN v4r manifest={fr['manifest']} gates={fr['gates']} git={fr['git']} science={fr['science']}")
    console.print(f"SEARCH_FEATURES={man['search_features']}")
    console.print(f"D1 {man['splits']['D1']} D2 {man['splits']['D2']} D3 {man['splits']['D3']}")
    console.print("SIGNAL_ONLY. Sealed OOS inaccessible. Zero survivors is acceptable. NO V4R2.")
    if not execute and not smoke:
        console.print("STOP after freeze print. Full run: campaign-v4r --execute")
        raise typer.Exit(code=0)
    state = run_campaign_v4r(root, smoke_days=["synthetic"] if smoke else None)
    console.print(f"report={state.get('report')} RESEARCH_PASS={state.get('RESEARCH_PASS')}")
    console.print("SIGNAL_ONLY. Sealed OOS not opened.")


@disc_app.command("campaign-v5")
def discovery_campaign_v5(
    execute: bool = typer.Option(False, "--execute", help="Forbidden until independent review. Default is readiness-only."),
) -> None:
    """V5 mechanism-first campaign. Default prints freeze identity and exits."""
    from quant_edge_lab.discovery.v5.runner import run_campaign_v5

    root = Path.cwd()
    state = run_campaign_v5(root, execute=execute)
    if not execute:
        console.print(f"FROZEN v5 manifest={state['manifest_hash']} gates={state['gates_hash']} git={state['git']}")
        console.print(f"data_manifest={state['data_manifest_hash']} science={state['science']}")
        console.print(f"campaigns={state['campaigns']}")
        console.print(f"partitions D1={state['partitions']['D1']} D2={state['partitions']['D2']} D3={state['partitions']['D3']}")
        console.print(f"primary_outcome={state['primary_outcome']} trial_count={state['trial_count']}")
        console.print(f"sealed_oos={state['sealed_oos']}")
        console.print(f"execution_status={state.get('execution_status')}")
        console.print(f"launch={state['launch_command']}")
        console.print("SIGNAL_ONLY. STOP. Default does not run research.")
        raise typer.Exit(code=0)
    console.print(json.dumps({k: state[k] for k in state if k != "rules"}, default=str)[:4000])
    console.print("SIGNAL_ONLY. Sealed OOS not opened.")


@disc_app.command("campaign-v5-continuation")
def discovery_campaign_v5_continuation(
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Forbidden until independent review. Default is readiness-only.",
    ),
) -> None:
    """Continuation confirmation campaign. Default prints readiness and does not load D3."""
    from quant_edge_lab.discovery.v5.continuation_runner import run_campaign_v5_continuation

    root = Path.cwd()
    if execute:
        try:
            state = run_campaign_v5_continuation(root, execute=True)
        except RuntimeError as exc:
            console.print(str(exc))
            raise typer.Exit(code=1) from exc
        console.print(json.dumps({k: state[k] for k in state if k != "rules"}, default=str)[:4000])
        console.print("SIGNAL_ONLY. Sealed OOS not opened. D1/D2 were not confirmation.")
        return
    state = run_campaign_v5_continuation(root, execute=False)
    console.print(
        "READINESS continuation "
        f"campaign_id={state['campaign_id']} science_id={state['science_id']}"
    )
    console.print(f"source_campaign={state['source_campaign']}")
    console.print(
        f"manifest={state['manifest_hash']} gates={state['gates_hash']} git={state['git']}"
    )
    console.print(f"data_manifest={state['data_manifest_hash']}")
    console.print(
        f"primary_outcome={state['primary_outcome']} trial_count={state['trial_count']}"
    )
    console.print(
        f"evaluable_split={state['evaluable_split']} "
        f"economic_floor_bp={state['economic_floor_bp']}"
    )
    console.print(f"sealed_oos={state['sealed_oos']}")
    console.print(f"execution_status={state['execution_status']}")
    console.print(f"h1_role={state['h1_role']}")
    console.print(f"hypotheses={state['hypotheses']}")
    console.print(
        f"z_threshold_abs={state['z_threshold_abs']} "
        f"scale_lookback={state['scale_lookback_sessions']} "
        f"scale_requires_all_sessions={state['scale_requires_all_sessions']}"
    )
    console.print(
        f"primary_estimand={state['primary_estimand']} "
        f"inference_unit={state['inference_unit']}"
    )
    console.print(f"bh_q={state['bh_q']} D3={state['d3_start']}..{state['d3_end']}")
    console.print(f"launch={state['launch_command']}")
    console.print("SIGNAL_ONLY. STOP. Default does not load D3 or run confirmation.")
    raise typer.Exit(code=0)


@disc_app.command("v5-status")
def discovery_v5_status(run_id: str | None = typer.Option(None, help="Optional run id. Never starts research.")) -> None:
    """Read-only V5 checkpoint/status. Does not mutate artifacts."""
    from quant_edge_lab.discovery.v5.status import summarize_status

    console.print(summarize_status(Path.cwd(), run_id))


@disc_app.command("campaign-v4")
def discovery_campaign_v4(
    execute: bool = typer.Option(False, "--execute", help="Do not use until freeze is accepted. Full matrix/D1–D3."),
) -> None:
    """Freeze V4 algorithm-discovery engine. Does not modify V2/V3. No sealed OOS."""
    from quant_edge_lab.discovery.campaign_v4 import freeze_campaign, load_v4, write_freeze_report

    root = Path.cwd()
    fr = freeze_campaign(root)
    man, gates = load_v4(root)
    path = write_freeze_report(root)
    console.print(f"FROZEN v4 manifest={fr['manifest']} gates={fr['gates']} git={fr['git']}")
    console.print(f"D1 {man['splits']['D1']} D2 {man['splits']['D2']} D3 {man['splits']['D3']}")
    console.print(f"primary_target={man['targets']['primary']} floor={gates['d3']['abs_mean_floor']} depth={gates['tree']['max_depth']} alpha={gates['tree']['node_alpha']}")
    console.print("SIGNAL_ONLY. Sealed OOS inaccessible. Zero survivors is acceptable.")
    console.print(f"freeze_report={path}")
    if not execute:
        console.print("STOP after freeze. Full V4 execution requires a separate --execute after review.")
        raise typer.Exit(code=0)
    from quant_edge_lab.discovery.campaign_v4_runner import run_campaign_v4

    state = run_campaign_v4(root)
    console.print(f"report={state.get('report')} RESEARCH_PASS={((state.get('counts') or {}).get('RESEARCH_PASS'))}")
    console.print("SIGNAL_ONLY. Sealed OOS not opened. Zero survivors is acceptable.")


@app.command("build-v4-features")
def build_v4_features(day: str = typer.Argument(..., help="YYYY-MM-DD causal_v2 day")) -> None:
    from quant_edge_lab.discovery.campaign_v4 import build_day_matrix

    feat = build_day_matrix(Path.cwd(), day)
    console.print(f"rows={feat.height} cols={len(feat.columns)} SIGNAL_ONLY")


@app.command("build-v4-peer-graphs")
def build_v4_peer_graphs() -> None:
    console.print("Peer graphs are built PIT at weekly as_of during V4 execute, not at freeze.")
    raise typer.Exit(code=0)


@app.command("run-v4-discovery")
def run_v4_discovery() -> None:
    console.print("D1 discovery runs only after freeze acceptance (--execute). Not launched.")
    raise typer.Exit(code=1)


@disc_app.command("status")
def discovery_status(batch_id: str = "smoke-executable") -> None:
    from quant_edge_lab.discovery.runner import load_state

    st = load_state(Path.cwd(), batch_id)
    if not st:
        console.print("No state for", batch_id)
        return
    console.print_json(data={k: st[k] for k in st if k != "panel"} | {"panel_n": len(st.get("panel") or [])})


@disc_app.command("resume")
def discovery_resume(batch_id: str, stage: str = "fast") -> None:
    from quant_edge_lab.discovery.runner import load_state, run_batch

    st = load_state(Path.cwd(), batch_id)
    ids = list((st.get("jobs") or {}).keys())
    run_batch(Path.cwd(), ids, batch_id=batch_id, stage=stage, resume=True)  # type: ignore[arg-type]


@know_app.command("search")
def knowledge_search(query: str) -> None:
    from quant_edge_lab.discovery.knowledge import import_experiment_001, search_knowledge

    import_experiment_001(Path.cwd())
    console.print_json(data=search_knowledge(Path.cwd(), query))


@know_app.command("history")
def knowledge_history(family_id: str) -> None:
    from quant_edge_lab.discovery.knowledge import knowledge_dir

    hits = []
    idx = knowledge_dir(Path.cwd()) / "index.jsonl"
    if idx.exists():
        for line in idx.read_text(encoding="utf-8").splitlines():
            if family_id in line:
                hits.append(json.loads(line))
    console.print_json(data=hits[-20:])
