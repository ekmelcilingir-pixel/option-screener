"""Vertical credit spreads — bull put in uptrends, bear call in downtrends.

Ideal profile: clear, persistent trend (price vs SMA50/200, slope), IV rank
high enough that the credit is meaningful, liquid short strike, no earnings
before expiry, and enough room between price and the short strike.
"""
from __future__ import annotations

import pandas as pd

from ..metrics import Features
from .base import Strategy, Candidate, ramp, peak


def _wing(df: pd.DataFrame, short_strike: float, width: float, side: str) -> pd.Series | None:
    target = short_strike - width if side == "put" else short_strike + width
    cand = df[df.strike < short_strike] if side == "put" else df[df.strike > short_strike]
    if cand.empty:
        return None
    return cand.loc[(cand.strike - target).abs().idxmin()]


class CreditSpread(Strategy):
    key = "spread"
    label = "Credit Spread (Bull Put / Bear Call)"
    description = "Defined-risk vertical: bull put in uptrends, bear call in downtrends; needs trend persistence, IV rank and liquid strikes."

    def evaluate(self, f: Features, cfg: dict) -> Candidate | None:
        t = f.tech
        if f.earnings_in_window:
            return "earnings before expiry"
        if t.trend == "flat":
            return "no trend"
        side = "put" if t.trend == "up" else "call"
        short = f.put30 if side == "put" else f.call30
        liq = f.put_liq if side == "put" else f.call_liq
        if short is None or not liq[0]:
            return "illiquid short strike"
        width = max(round(t.price * cfg["options"]["spread_width_pct"]), 1)
        return self._score(f, side, short, liq, width)

    def _score(self, f: Features, side: str, short: pd.Series, liq: tuple, width: float) -> Candidate | None:
        t = f.tech
        # Long wing from the full chain when available; otherwise a conservative estimate
        # (wing ≈ 35% of the short-leg credit for a ~5%-wide spread).
        wing = None
        if f.chain is not None:
            df = f.chain.puts if side == "put" else f.chain.calls
            wing = _wing(df, float(short.strike), width, side)
        credit_short = float(short.mid)
        credit_long = float(wing.mid) if wing is not None else credit_short * 0.35
        long_strike = float(wing.strike) if wing is not None else (short.strike - width if side == "put" else short.strike + width)
        net = credit_short - credit_long
        w = abs(float(short.strike) - long_strike)
        if w <= 0 or net <= 0:
            return "no net credit"
        max_loss = w - net
        roc = net / max_loss if max_loss > 0 else float("nan")

        maxp = {"trend_strength": 30, "iv_rank": 20, "cushion": 15, "liquidity": 15, "reward_risk": 15, "not_extended": 5}
        fac, notes = {}, []
        slope = t.sma50_slope if side == "put" else -t.sma50_slope
        dist = t.dist_sma200 if side == "put" else -t.dist_sma200
        r60 = t.ret_60d if side == "put" else -t.ret_60d
        fac["trend_strength"] = ramp(slope, 0, 0.05, 12) + ramp(dist, 0, 0.15, 10) + ramp(r60, 0, 0.15, 8)
        fac["iv_rank"] = ramp(f.iv_rank, 15, 65, 20)
        cushion = (1 - short.strike / t.price) if side == "put" else (short.strike / t.price - 1)
        fac["cushion"] = ramp(cushion, 0.03, 0.10, 15)
        _, spread_pct, oi = liq
        fac["liquidity"] = ramp(spread_pct, 0.30, 0.05, 9) + ramp(oi, 500, 5000, 6)
        fac["reward_risk"] = peak(roc, 0.10, 0.25, 0.45, 0.80, 15)
        rsi_ok = t.rsi14 if side == "put" else 100 - t.rsi14
        fac["not_extended"] = peak(rsi_ok, 30, 40, 65, 78, 5)
        if (side == "put" and t.rsi14 > 72) or (side == "call" and t.rsi14 < 28):
            notes.append("extended — wait for mean reversion")

        setup = {
            "type": "bull_put" if side == "put" else "bear_call", "expiry": f.expiry, "dte": f.dte,
            "short_strike": float(short.strike), "long_strike": long_strike, "short_delta": round(float(short.delta), 2),
            "net_credit": round(net, 2), "width": round(w, 2), "max_loss": round(max_loss, 2),
            "return_on_risk": round(roc, 3), "cushion_pct": round(cushion, 3),
            "wing_source": "chain" if wing is not None else "estimated",
        }
        if f.iv_method == "proxy":
            notes.append("IV rank via HV proxy")
        return self.finish(self, f, fac, maxp, setup, notes)
