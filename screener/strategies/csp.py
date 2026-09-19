"""Cash-secured put / Wheel entry.

Ideal profile: a quality name you are happy to own, in an uptrend or a pullback
to support, with IV elevated versus its own history and versus realised vol,
a liquid ~30-delta put, and no earnings before expiry.
"""
from __future__ import annotations

from ..metrics import Features
from .base import Strategy, Candidate, ramp, peak, quality_points, annualised_yield


class CashSecuredPut(Strategy):
    key = "csp"
    label = "Cash-Secured Put / Wheel"
    description = "Sell ~30Δ put on quality names with elevated IV, uptrend or pullback to support, no earnings before expiry."

    def evaluate(self, f: Features, cfg: dict) -> Candidate | None:
        t = f.tech
        if f.put30 is None or not f.put_liq[0]:
            return "illiquid 30Δ put"
        if f.earnings_in_window:
            return "earnings before expiry"
        if t.trend == "down" or t.dist_sma200 < -0.10:
            return "downtrend / >10% below SMA200"

        maxp = {"iv_rank": 25, "iv_vs_hv": 15, "trend": 20, "pullback": 10, "quality": 15, "liquidity": 10, "yield": 5}
        fac, notes = {}, []
        fac["iv_rank"] = ramp(f.iv_rank, 20, 70, 25)
        fac["iv_vs_hv"] = peak(f.iv_hv20, 0.8, 1.15, 1.8, 3.0, 15)
        fac["trend"] = (12 if t.trend == "up" else 6) + ramp(t.sma50_slope, 0, 0.04, 8)
        # Best entries: price 0..-8% below SMA20 but above SMA50/200 (pullback), RSI 35-55
        fac["pullback"] = peak(t.rsi14, 25, 35, 55, 70, 6) + peak(t.dist_sma50, -0.08, -0.04, 0.03, 0.10, 4)
        q, qn = quality_points(f, 15)
        fac["quality"] = q
        notes += qn
        _, spread_pct, oi = f.put_liq
        fac["liquidity"] = ramp(spread_pct, 0.30, 0.05, 6) + ramp(oi, 500, 5000, 4)

        p = f.put30
        credit = float(p.mid)
        strike = float(p.strike)
        ann = annualised_yield(credit, strike, f.dte)
        fac["yield"] = ramp(ann, 0.10, 0.30, 5)
        setup = {
            "expiry": f.expiry, "dte": f.dte, "strike": strike, "delta": round(float(p.delta), 2),
            "credit": round(credit, 2), "cushion_pct": round(1 - strike / t.price, 3),
            "breakeven": round(strike - credit, 2), "annualised_yield": round(ann, 3),
            "capital_per_contract": round(strike * 100, 0),
        }
        if f.iv_method == "proxy":
            notes.append("IV rank via HV proxy")
        if f.days_to_earnings is not None and f.days_to_earnings <= f.dte + 21:
            notes.append(f"earnings in {f.days_to_earnings}d (after expiry)")
        return self.finish(self, f, fac, maxp, setup, notes)
