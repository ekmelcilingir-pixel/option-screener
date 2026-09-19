"""Earnings plays — implied vs realised move.

If the market prices a move (ATM straddle) well above what the stock has
historically delivered, premium selling (iron condor / short strangle) is
favoured. If implied is well below history, long straddle/strangle is favoured.
Requires an earnings date inside the window and a liquid chain.
"""
from __future__ import annotations

import numpy as np

from ..metrics import Features
from .base import Strategy, Candidate, ramp, peak


class EarningsPlay(Strategy):
    key = "earnings"
    label = "Earnings Play (Straddle / Strangle / Iron Condor)"
    description = "Compare straddle-implied move with realised moves over the last 8 reports; sell premium when rich, buy when cheap."

    def evaluate(self, f: Features, cfg: dict) -> Candidate | None:
        if f.days_to_earnings is None or not (0 <= f.days_to_earnings <= cfg["earnings"]["window_days"]):
            return "no earnings in window"
        if f.event_dte is None:
            return "no post-earnings expiry"
        if not (f.event_put_liq[0] and f.event_call_liq[0]):
            return "illiquid event chain"
        if len(f.realised_moves) < 4:
            return "insufficient earnings history"

        hist = np.array(f.realised_moves)
        avg, med, mx = hist.mean(), np.median(hist), hist.max()
        # Straddle price ≈ 0.8·σ·√T. Back out total implied variance for the expiry, strip the
        # variance of the ordinary (non-event) trading days at realised vol, and convert the
        # remaining event variance back to an expected absolute move.
        trading_days = max(round(f.event_dte * 252 / 365), 1)
        sigma_total_sq = (f.implied_move / 0.8) ** 2               # σ²·T for the expiry
        ordinary_var = (f.tech.hv60 ** 2) / 252 * max(trading_days - 1, 0)
        event_var = max(sigma_total_sq - ordinary_var, 0.0)
        event_implied = float(0.8 * np.sqrt(event_var))
        ratio = event_implied / avg if avg > 0 else float("nan")

        if ratio >= 1.15:
            bias, structure = "sell_premium", "iron_condor_or_short_strangle"
        elif ratio <= 0.85:
            bias, structure = "buy_vol", "long_straddle_or_strangle"
        else:
            bias, structure = "neutral", "no_edge"

        maxp = {"edge": 40, "consistency": 20, "liquidity": 20, "timing": 10, "history_depth": 10}
        fac, notes = {}, []
        edge = abs(ratio - 1)
        fac["edge"] = ramp(edge, 0.10, 0.60, 40)
        cv = hist.std() / avg if avg > 0 else 9
        fac["consistency"] = ramp(cv, 1.0, 0.3, 20)          # more consistent history -> more trust
        sp = max(f.event_put_liq[1], f.event_call_liq[1])
        oi = min(f.event_put_liq[2], f.event_call_liq[2])
        fac["liquidity"] = ramp(sp, 0.30, 0.05, 12) + ramp(oi, 500, 5000, 8)
        fac["timing"] = peak(f.days_to_earnings, -1, 1, 5, 14, 10)   # best to act 1-5 days before
        fac["history_depth"] = ramp(len(hist), 4, 8, 10)
        if bias == "neutral":
            fac["edge"] = 0
            notes.append("implied ≈ realised — no edge")
        if mx > 2.5 * avg:
            notes.append(f"tail move {mx:.0%} in history")

        setup = {
            "earnings_date": f.next_earnings.isoformat() if f.next_earnings else None, "days_to_earnings": f.days_to_earnings,
            "expiry": f.event_expiry, "implied_move_expiry": round(f.implied_move, 3), "implied_move_event": round(event_implied, 3),
            "realised_avg": round(float(avg), 3), "realised_median": round(float(med), 3), "realised_max": round(float(mx), 3),
            "implied_vs_realised": round(ratio, 2), "bias": bias, "structure": structure, "n_events": int(len(hist)),
        }
        return self.finish(self, f, fac, maxp, setup, notes)
