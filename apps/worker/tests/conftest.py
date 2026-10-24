import os


def latency_budget(seconds: float) -> float:
    """A wall-clock limit that stays meaningful on a busy machine: scaled by how many runnable
    processes compete for each core (tests run next to model downloads and other jobs), and by
    TERJEMAHIN_LATENCY_SCALE for slow CI runners."""
    try:
        load = os.getloadavg()[0] / (os.cpu_count() or 1)
    except OSError:
        load = 0.0
    return seconds * max(1.0, 1.0 + load) * float(os.environ.get("TERJEMAHIN_LATENCY_SCALE", "1"))
