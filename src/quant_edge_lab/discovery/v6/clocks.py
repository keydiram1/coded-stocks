"""P0/P1/P2, T1+2 entry, TOD and |z| buckets. No I/O."""

from __future__ import annotations

from datetime import datetime, time, timedelta

IMPULSE_START_MIN = time(10, 0)
IMPULSE_END_MAX = time(14, 0)
WAIT_END_MAX = time(14, 30)
T1_BUCKETS = (
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
)
Z_BUCKETS = ((2.0, 2.5, False), (2.5, 3.0, False), (3.0, float("inf"), True))


def _hhmm(t: time) -> int:
    return t.hour * 60 + t.minute


def add_minutes(t: time, minutes: int) -> time:
    total = _hhmm(t) + minutes
    return time(total // 60, total % 60)


def impulse_window_ok(start: time, w_imp: int) -> bool:
    if start < IMPULSE_START_MIN:
        return False
    end = add_minutes(start, w_imp - 1)
    return end <= IMPULSE_END_MAX


def wait_end(start: time, w_imp: int, w_wait: int) -> time:
    return add_minutes(start, w_imp + w_wait - 1)


def wait_ok(start: time, w_imp: int, w_wait: int) -> bool:
    return wait_end(start, w_imp, w_wait) <= WAIT_END_MAX


def p0_time(impulse_start: time) -> time:
    return add_minutes(impulse_start, -1)


def p1_time(impulse_start: time, w_imp: int) -> time:
    return add_minutes(impulse_start, w_imp - 1)


def p2_time(impulse_start: time, w_imp: int, w_wait: int) -> time:
    return wait_end(impulse_start, w_imp, w_wait)


def t1_plus_k(t1: time, k: int) -> time:
    return add_minutes(t1, k)


def entry_time(t1: time) -> time:
    """Primary research entry = T1+2 open. Skip T1+1 open."""
    return t1_plus_k(t1, 2)


def outcome_close_time(entry: time, horizon_minutes: int) -> time:
    return add_minutes(entry, horizon_minutes - 1)


def t1_bucket(t1: time) -> str | None:
    m = _hhmm(t1)
    start = (m // 30) * 30
    h, mm = divmod(start, 60)
    label = f"{h:02d}:{mm:02d}"
    return label if label in T1_BUCKETS else None


def abs_z_bucket(abs_z: float) -> str | None:
    z = float(abs_z)
    if not np_finite(z) or z < 2.0:
        return None
    for lo, hi, right_incl in Z_BUCKETS:
        if z >= lo and (z <= hi if right_incl else z < hi):
            return f"[{lo},{hi}{']' if right_incl else ')'}"
    return None


def np_finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def signal_available_at(t1_bar_ts: datetime) -> datetime:
    return t1_bar_ts + timedelta(minutes=1)


def decision_ts_entry(t1_bar_ts: datetime) -> datetime:
    """T1+2 open. T1 bar [t,t+1); close known at t+1; skip that open; enter t+2."""
    return t1_bar_ts + timedelta(minutes=2)
