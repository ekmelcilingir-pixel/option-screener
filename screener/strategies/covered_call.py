"""Covered call (or buy-write) income.

Ideal profile: stable, range-bound-to-mildly-bullish name, IV rich relative to
realised, not in a parabolic run (RSI not > 70) so the upside cap is unlikely to
hurt, liquid ~30Δ call, dividend a bonus, no earnings before expiry.
"""
from __future__ import annotations

from ..metrics import Features
from .base import Strategy, Candidate, ramp, peak, quality_points, annualised_yield


class CoveredCall(Strategy):
    key = "cc"
    label = "Covered Call / Buy-Write"
    description = "Sell ~30Δ call against stock on stable names with rich IV, moderate momentum, no earnings before expiry."

    def evaluate(self, f: Features, cfg: dict) -> Candidate | None:
        t = f.tech
        if f.call30 is None or not f.call_liq[0]:
            return "illiquid 30Δ call"
        if f.earnings_in_window:
            return "earnings before expiry"
        if t.trend == "down" and t.dist_sma200 < -0.05:
            return "downtrend"

        maxp = {"iv_rank": 20, "iv_vs_hv": 20, "stability": 20, "momentum": 15, "quality": 10, "liquidity": 10, "yield": 5}
        fac, notes = {}, []
        fac["iv_rank"] = ramp(f.iv_rank, 15, 65, 20)
        fac["iv_vs_hv"] = ramp(f.iv_hv60, 0.9, 1.6, 20)
        # Stability: lower absolute HV, low beta, modest drawdown
        beta = (f.info or {}).get("beta") or 1.0
        fac["stability"] = ramp(t.hv60, 0.60, 0.20, 10) + peak(beta, 0.2, 0.5, 1.1, 2.0, 5) + ramp(t.drawdown_60d, -0.20, -0.03, 5)
        # Momentum: mildly positive, not overbought (cap risk) and not collapsing
        fac["momentum"] = peak(t.rsi14, 30, 42, 62, 72, 8) + peak(t.ret_20d, -0.10, -0.02, 0.05, 0.15, 7)
        q, qn = quality_points(f, 10)
        fac["quality"] = q
        notes += qn
        _, spread_pct, oi = f.call_liq
        fac["liquidity"] = ramp(spread_pct, 0.30, 0.05, 6) + ramp(oi, 500, 5000, 4)

        c = f.call30
        credit = float(c.mid)
        strike = float(c.strike)
        ann = annualised_yield(credit, t.price, f.dte)
        fac["yield"] = ramp(ann, 0.08, 0.25, 5)
        dy = (f.info or {}).get("dividendYield") or 0
        dy = dy / 100                      # yfinance >=0.2.5x reports dividendYield in percent (0.46 == 0.46%)
        if dy > 0.015:
            notes.append(f"div yield {dy:.1%}")
        setup = {
            "expiry": f.expiry, "dte": f.dte, "strike": strike, "delta": round(float(c.delta), 2),
            "credit": round(credit, 2), "upside_to_strike_pct": round(strike / t.price - 1, 3),
            "annualised_yield": round(ann, 3), "downside_breakeven": round(t.price - credit, 2),
        }
        if f.iv_method == "proxy":
            notes.append("IV rank via HV proxy")
        return self.finish(self, f, fac, maxp, setup, notes)
