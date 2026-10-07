"""Periodic scientific/runtime status. Not a per-row logger."""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console

console = Console()
_LAST: dict[str, float] = {}
_INTERVAL = 12 * 60


def maybe_status(key: str, body: str, *, force: bool = False, interval: float = _INTERVAL) -> None:
    now = time.time()
    prev = _LAST.get(key, 0.0)
    if not force and (now - prev) < interval:
        return
    _LAST[key] = now
    console.print(body)


def fmt_elapsed(t0: float) -> str:
    s = int(max(time.time() - t0, 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def graph_status(*, done: int, total: int, as_of: str, lin: dict[str, Any], edge_hist: list[int], peak_rss_gb: float, t0: float, zeros: int) -> str:
    med = sorted(edge_hist)[len(edge_hist) // 2] if edge_hist else 0
    return (
        "[V4 GRAPH STATUS]\n"
        f"weeks: {done} / {total}\n"
        f"current_as_of: {as_of}\n"
        f"names_this_week: {lin.get('n_names')}\n"
        f"usable_names: {lin.get('n_usable_names')}\n"
        f"structural_pairs_considered: {lin.get('structural_pairs_considered')}\n"
        f"lag_tests: {lin.get('n_tests')}\n"
        f"BH_survivors: {lin.get('n_bh_survivors')}\n"
        f"final_edges: {lin.get('n_edges')}\n"
        f"median_edges_per_completed_graph: {med}\n"
        f"zero_edge_graphs: {zeros} / {done}\n"
        f"median_abs_r: {lin.get('median_abs_r')}\n"
        f"peak_RSS: {peak_rss_gb:.2f} GB\n"
        f"elapsed: {fmt_elapsed(t0)}"
    )
