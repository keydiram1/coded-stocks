from pathlib import Path

from quant_edge_lab.data.sample import write_sample_data
from quant_edge_lab.hypotheses.loader import hypothesis_hash, load_hypothesis
from quant_edge_lab.pipeline import reproduce_experiment, run_hypothesis


def test_experiment_hash_stable_for_same_spec():
    spec = load_hypothesis(Path("hypotheses/gap_rvol_continuation_v1.yaml"))
    assert hypothesis_hash(spec) == hypothesis_hash(spec)


def test_end_to_end_sample_and_reproduce(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    src = Path(__file__).resolve().parents[2]
    import shutil

    shutil.copytree(src / "hypotheses", tmp_path / "hypotheses")
    shutil.copytree(src / "config", tmp_path / "config")
    write_sample_data(tmp_path, seed=42)
    rec = run_hypothesis(tmp_path / "hypotheses" / "gap_rvol_continuation_v1.yaml", root=tmp_path)
    assert rec.stage == "EXPLORATORY"
    assert Path(rec.events_path).exists()
    assert Path(rec.outcomes_path).exists()
    assert Path(rec.summary_path).exists()
    assert Path(rec.report_path).exists()
    assert rec.events_hash
    assert rec.hypothesis_hash
    result = reproduce_experiment(rec.experiment_id, root=tmp_path)
    assert result["match"] is True


def test_all_three_hypotheses_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import shutil

    src = Path(__file__).resolve().parents[2]
    shutil.copytree(src / "hypotheses", tmp_path / "hypotheses")
    shutil.copytree(src / "config", tmp_path / "config")
    write_sample_data(tmp_path, seed=42)
    for name in (
        "gap_rvol_continuation_v1.yaml",
        "extreme_extension_fade_v1.yaml",
        "premarket_high_behavior_v1.yaml",
    ):
        rec = run_hypothesis(tmp_path / "hypotheses" / name, root=tmp_path)
        assert rec.experiment_id
