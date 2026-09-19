"""Strategy scoring framework.

Each strategy converts a Features bundle into a Candidate with a 0-100 score
and a transparent factor breakdown. Hard filters return None (disqualified).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..metrics import Features


@dataclass
class Candidate:
    strategy: str
    symbol: str
    name: str
    sector: str
    score: float
    factors: dict[str, float]          # factor -> points awarded
    max_points: dict[str, float]       # factor -> maximum possible
    setup: dict                        # strategy-specific trade sketch (strike, credit, yield...)
    notes: list[str] = field(default_factory=list)
    features: Features | None = None


def ramp(x: float, lo: float, hi: float, pts: float) -> float:
    """Linear 0..pts as x goes lo..hi (clipped). Reverse by passing lo > hi."""
    if np.isnan(x):
        return 0.0
    if lo == hi:
        return pts if x >= hi else 0.0
    return float(pts * np.clip((x - lo) / (hi - lo), 0, 1))


def peak(x: float, lo: float, best_lo: float, best_hi: float, hi: float, pts: float) -> float:
    """Trapezoid: 0 at lo, full pts between best_lo..best_hi, 0 at hi."""
    if np.isnan(x):
        return 0.0
    if x <= lo or x >= hi:
        return 0.0
    if best_lo <= x <= best_hi:
        return pts
    if x < best_lo:
        return pts * (x - lo) / (best_lo - lo)
    return pts * (hi - x) / (hi - best_hi)


class Strategy(ABC):
    key: str
    label: str
    description: str

    @abstractmethod
    def evaluate(self, f: Features, cfg: dict) -> Candidate | str | None:
        """Return a Candidate, or a short string naming the hard filter that rejected the name."""

    @staticmethod
    def finish(strategy: "Strategy", f: Features, factors: dict, maxp: dict, setup: dict, notes: list[str]) -> Candidate:
        score = 100 * sum(factors.values()) / sum(maxp.values())
        return Candidate(strategy.key, f.symbol, f.name, f.sector, round(score, 1), factors, maxp, setup, notes, f)


def quality_points(f: Features, pts: float) -> tuple[float, list[str]]:
    """Fundamental sanity: size, profitability, valuation not absurd, leverage not extreme."""
    i = f.info or {}
    notes, p = [], 0.0
    mc = i.get("marketCap") or 0
    p += ramp(mc, 5e9, 5e10, pts * 0.3)
    pm = i.get("profitMargins")
    if pm is not None:
        p += ramp(pm, 0.0, 0.15, pts * 0.3)
        if pm < 0:
            notes.append("unprofitable")
    fpe = i.get("forwardPE")
    if fpe is not None and fpe > 0:
        p += peak(fpe, 0, 8, 30, 60, pts * 0.25)
        if fpe > 50:
            notes.append(f"fwd P/E {fpe:.0f}")
    de = i.get("debtToEquity")
    if de is not None:
        p += ramp(de, 250, 50, pts * 0.15)
    else:
        p += pts * 0.075
    return min(p, pts), notes


def annualised_yield(premium: float, capital: float, dte: int) -> float:
    return premium / capital * 365 / max(dte, 1) if capital else float("nan")
