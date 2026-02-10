# brain/strategy_hash.py
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def genome_hash(genome: Dict[str, Any]) -> str:
    """
    Stable hash of genome dict (content-based).
    """
    payload = json.dumps(genome or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
