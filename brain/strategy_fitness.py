# brain/strategy_fitness.py

def compute_fitness(stats: dict) -> float:
    """
    Compute fitness score from outcome stats.
    Expected keys:
      - avg_pnl (float)
      - win_rate (float 0..1)
      - drawdown (float >= 0)
      - samples (int)
    """
    avg_pnl = float(stats.get("avg_pnl", 0.0))
    win_rate = float(stats.get("win_rate", 0.0))
    drawdown = float(stats.get("drawdown", 0.0))
    samples = int(stats.get("samples", 0))

    # Weights (conservative v1)
    W_PNL = 1.0
    W_WIN = 0.2
    W_DD = 0.5

    # Penalize small sample size
    if samples <= 0:
        sample_penalty = 1.0
    elif samples < 10:
        sample_penalty = 0.5
    elif samples < 30:
        sample_penalty = 0.2
    else:
        sample_penalty = 0.0

    fitness = (
        avg_pnl * W_PNL
        + win_rate * W_WIN
        - drawdown * W_DD
        - sample_penalty
    )

    return float(fitness)
