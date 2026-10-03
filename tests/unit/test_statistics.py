import numpy as np

from quant_edge_lab.statistics.summarize import _block_mean_distribution, benjamini_hochberg


def test_bh_rejects_small_p_controls_fdr():
    p = [0.001, 0.02, 0.9, 0.8]
    res = benjamini_hochberg(p, q=0.05)
    assert res[0]["bh_rejected"] is True
    assert res[2]["bh_rejected"] is False


def test_bootstrap_deterministic_seed():
    values = np.array([0.01, -0.02, 0.03, 0.00, 0.02, -0.01])
    blocks = np.array(["a", "a", "b", "b", "c", "c"])
    a = _block_mean_distribution(values, blocks, n_boot=200, seed=42)
    b = _block_mean_distribution(values, blocks, n_boot=200, seed=42)
    c = _block_mean_distribution(values, blocks, n_boot=200, seed=7)
    assert np.allclose(a, b)
    assert not np.allclose(a, c)
