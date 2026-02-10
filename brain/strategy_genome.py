# brain/strategy_genome.py
from __future__ import annotations

import random
from copy import deepcopy
from typing import Dict, Any

# Ranges are intentionally conservative for v1 (easy to tune later)
GENOME_SCHEMA = {
    "entry_threshold": (0.10, 0.90),  # score threshold to enter
    "sl_atr_mult": (0.5, 5.0),
    "tp_atr_mult": (0.5, 8.0),
    "risk_per_trade": (0.001, 0.03),  # 0.1% to 3%
    "only_trend": (0, 1),             # bool encoded as 0/1
    "avoid_high_vol": (0, 1),         # bool encoded as 0/1
}

NUM_KEYS = {"entry_threshold", "sl_atr_mult", "tp_atr_mult", "risk_per_trade"}
BOOL_KEYS = {"only_trend", "avoid_high_vol"}


def _clamp(v: float, lo: float, hi: float) -> float:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def clamp_genome(genome: Dict[str, Any]) -> Dict[str, Any]:
    """Return a clamped copy of genome so every value stays inside schema ranges."""
    g = deepcopy(genome)
    for k, (lo, hi) in GENOME_SCHEMA.items():
        if k not in g:
            continue
        if k in BOOL_KEYS:
            g[k] = 1 if int(g[k]) == 1 else 0
        else:
            g[k] = float(_clamp(float(g[k]), float(lo), float(hi)))
    return g


def random_genome(rng: random.Random | None = None) -> Dict[str, Any]:
    """Create a valid random genome."""
    rng = rng or random.Random()
    g: Dict[str, Any] = {}
    for k, (lo, hi) in GENOME_SCHEMA.items():
        if k in BOOL_KEYS:
            g[k] = rng.choice([0, 1])
        else:
            g[k] = rng.uniform(lo, hi)
    return clamp_genome(g)


def mutate(genome: Dict[str, Any], rate: float = 0.10, rng: random.Random | None = None) -> Dict[str, Any]:
    """
    Mutate genome with a per-key probability = rate.
    - Numeric keys: add small gaussian noise proportional to range
    - Bool keys: flip
    Returns a new mutated genome (does not modify input).
    """
    rng = rng or random.Random()
    base = clamp_genome(genome)
    g = deepcopy(base)

    for k, (lo, hi) in GENOME_SCHEMA.items():
        if rng.random() > rate:
            continue

        if k in BOOL_KEYS:
            g[k] = 0 if int(g[k]) == 1 else 1
        else:
            span = float(hi) - float(lo)
            noise = rng.gauss(0.0, 0.10 * span)  # 10% of range std
            g[k] = float(g[k]) + noise

    return clamp_genome(g)
