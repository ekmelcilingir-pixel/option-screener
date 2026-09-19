"""Derived metrics: volatility, technicals, option-chain analytics, IV rank store."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .market_data import SymbolData, ChainSnapshot

log = logging.getLogger(__name__)
IV_STORE = Path("data/iv_history.csv")


# --------------------------------------------------------------------------- Black-Scholes helpers
def _N(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(kind: str, s: float, k: float, t: float, r: float, iv: float) -> float:
    if t <= 0 or iv <= 0 or s <= 0 or k <= 0:
        return float("nan")
    d1 = (math.log(s / k) + (r + iv * iv / 2) * t) / (iv * math.sqrt(t))
    return _N(d1) if kind == "call" else _N(d1) - 1


def bs_price(kind: str, s: float, k: float, t: float, r: float, iv: float) -> float:
    d1 = (math.log(s / k) + (r + iv * iv / 2) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    if kind == "call":
        return s * _N(d1) - k * math.exp(-r * t) * _N(d2)
    return k * math.exp(-r * t) * _N(-d2) - s * _N(-d1)


def implied_vol(kind: str, price: float, s: float, k: float, t: float, r: float) -> float:
    """Bisection IV solver; NaN when the price sits outside no-arbitrage bounds."""
    intrinsic = max(0.0, (s - k * math.exp(-r * t)) if kind == "call" else (k * math.exp(-r * t) - s))
    if t <= 0 or price <= intrinsic + 1e-6 or price >= (s if kind == "call" else k):
        return float("nan")
    lo, hi = 0.01, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if bs_price(kind, s, k, t, r, mid) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _fill_iv(df: pd.DataFrame, kind: str, spot: float, t: float, r: float) -> None:
    """Back-solve missing IVs from mid; interpolate across strikes for anything still missing."""
    if df.empty:
        return
    miss = df.iv.isna()
    if miss.any():
        df.loc[miss, "iv"] = [implied_vol(kind, m, spot, k, t, r) for m, k in zip(df.loc[miss, "mid"], df.loc[miss, "strike"])]
    df["iv"] = df.set_index("strike").iv.interpolate(method="index", limit_direction="both").values
    df["iv_source"] = np.where(miss, "solved", "yahoo")


def add_deltas(chain: ChainSnapshot, spot: float, r: float) -> None:
    t = chain.dte / 365
    _fill_iv(chain.calls, "call", spot, t, r)
    _fill_iv(chain.puts, "put", spot, t, r)
    chain.calls["delta"] = [bs_delta("call", spot, k, t, r, iv) for k, iv in zip(chain.calls.strike, chain.calls.iv)]
    chain.puts["delta"] = [bs_delta("put", spot, k, t, r, iv) for k, iv in zip(chain.puts.strike, chain.puts.iv)]
    chain.calls.dropna(subset=["iv", "delta"], inplace=True)
    chain.puts.dropna(subset=["iv", "delta"], inplace=True)


def pick_by_delta(df: pd.DataFrame, target_abs_delta: float, band: float = 0.10) -> pd.Series | None:
    """Strike near the target delta, preferring open interest: among strikes within ±band of the
    target |Δ| pick the one with the highest OI (liquidity clusters at round strikes); if none
    carry OI, fall back to the nearest delta."""
    if df is None or df.empty or "delta" not in df:
        return None
    ad = df.delta.abs()
    near = df[(ad >= target_abs_delta - band) & (ad <= target_abs_delta + band)]
    if not near.empty and near.oi.max() > 0:
        top = near[near.oi >= 0.5 * near.oi.max()]           # among the well-traded strikes...
        return top.loc[(top.delta.abs() - target_abs_delta).abs().idxmin()]   # ...closest to target
    return df.loc[(ad - target_abs_delta).abs().idxmin()]


def atm_iv(chain: ChainSnapshot, spot: float) -> float:
    c = chain.calls.iloc[(chain.calls.strike - spot).abs().argsort()[:2]]
    p = chain.puts.iloc[(chain.puts.strike - spot).abs().argsort()[:2]]
    return float(pd.concat([c.iv, p.iv]).dropna().mean())


def straddle_implied_move(chain: ChainSnapshot, spot: float) -> float:
    """ATM straddle mid / spot -> implied % move by expiry."""
    c = chain.calls.iloc[(chain.calls.strike - spot).abs().argmin()]
    p = chain.puts.iloc[(chain.puts.strike - spot).abs().argmin()]
    return float((c.mid + p.mid) / spot)


# --------------------------------------------------------------------------- technicals
@dataclass
class Technicals:
    price: float
    sma20: float
    sma50: float
    sma200: float
    rsi14: float
    hv20: float
    hv60: float
    hv252: float
    ret_20d: float
    ret_60d: float
    dist_sma50: float          # (price/sma50 - 1)
    dist_sma200: float
    sma50_slope: float         # 20-day % change of SMA50
    avg_dollar_vol: float
    drawdown_60d: float        # from 60d high
    trend: str                 # "up" | "down" | "flat"


def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean()
    return float(100 - 100 / (1 + rs.iloc[-1]))


def _hv(close: pd.Series, n: int) -> float:
    return float(np.log(close).diff().tail(n).std() * math.sqrt(252))


def technicals(prices: pd.DataFrame) -> Technicals:
    c = prices.close
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    price = float(c.iloc[-1])
    slope = float(sma50.iloc[-1] / sma50.iloc[-21] - 1) if len(sma50.dropna()) > 21 else 0.0
    up = price > sma50.iloc[-1] > sma200.iloc[-1] and slope > 0
    down = price < sma50.iloc[-1] < sma200.iloc[-1] and slope < 0
    return Technicals(
        price=price, sma20=float(sma20.iloc[-1]), sma50=float(sma50.iloc[-1]), sma200=float(sma200.iloc[-1]),
        rsi14=_rsi(c), hv20=_hv(c, 20), hv60=_hv(c, 60), hv252=_hv(c, 252),
        ret_20d=float(price / c.iloc[-21] - 1), ret_60d=float(price / c.iloc[-61] - 1),
        dist_sma50=float(price / sma50.iloc[-1] - 1), dist_sma200=float(price / sma200.iloc[-1] - 1),
        sma50_slope=slope, avg_dollar_vol=float((c * prices.volume).tail(20).mean()),
        drawdown_60d=float(price / c.tail(60).max() - 1),
        trend="up" if up else "down" if down else "flat",
    )


def realised_earnings_moves(prices: pd.DataFrame, events: list[date]) -> list[float]:
    """Absolute close-to-close move across each past earnings date (event day -> next session)."""
    moves = []
    idx = prices.index.tz_localize(None) if getattr(prices.index, "tz", None) else prices.index
    c = pd.Series(prices.close.values, index=pd.to_datetime(idx).date)
    dates = list(c.index)
    for ev in events:
        pos = np.searchsorted(dates, ev)
        if 1 <= pos < len(dates) - 1:
            before, after = c.iloc[pos - 1], c.iloc[pos + 1]   # spans AMC or BMO reporting
            moves.append(abs(after / before - 1))
    return moves


# --------------------------------------------------------------------------- IV rank store
def load_iv_store() -> pd.DataFrame:
    if IV_STORE.exists():
        df = pd.read_csv(IV_STORE, parse_dates=["date"])
        return df
    return pd.DataFrame(columns=["date", "symbol", "atm_iv", "hv20", "hv60", "price"])


def append_iv_store(rows: list[dict], today: date) -> None:
    df = load_iv_store()
    new = pd.DataFrame(rows)
    new["date"] = pd.Timestamp(today)
    if len(df):
        df = df[df.date.dt.date != today]
        out = pd.concat([df, new], ignore_index=True)
    else:
        out = new
    out = out.sort_values(["symbol", "date"])
    out.to_csv(IV_STORE, index=False)
    log.info("iv store: %d rows total", len(out))


def iv_rank_percentile(store: pd.DataFrame, symbol: str, iv_now: float, tech: Technicals, cfg: dict) -> tuple[float, float, str]:
    """Returns (iv_rank, iv_percentile, method). Falls back to an HV-based proxy with short history."""
    hist = store[store.symbol == symbol].atm_iv.dropna() if len(store) else pd.Series(dtype=float)
    hist = hist.tail(cfg["iv_rank"]["lookback_days"])
    if len(hist) >= cfg["iv_rank"]["min_history_days"]:
        lo, hi = hist.min(), hist.max()
        rank = 100 * (iv_now - lo) / (hi - lo) if hi > lo else 50.0
        pct = 100 * (hist < iv_now).mean()
        return float(rank), float(pct), f"iv_history({len(hist)}d)"
    # Proxy: where does current IV sit relative to the distribution of rolling 20d realised vol over 1y?
    # Reasonable stand-in until enough IV history has been collected.
    return float("nan"), float("nan"), "proxy"


def proxy_iv_rank(prices: pd.DataFrame, iv_now: float) -> tuple[float, float]:
    lr = np.log(prices.close).diff()
    rv = (lr.rolling(20).std() * math.sqrt(252)).dropna().tail(252)
    if rv.empty:
        return 50.0, 50.0
    lo, hi = rv.min(), rv.max()
    rank = 100 * (iv_now - lo) / (hi - lo) if hi > lo else 50.0
    pct = 100 * (rv < iv_now).mean()
    return float(np.clip(rank, 0, 100)), float(pct)


# --------------------------------------------------------------------------- liquidity
def liquidity_ok(row: pd.Series | None, cfg: dict) -> tuple[bool, float, int]:
    if row is None:
        return False, float("nan"), 0
    oi_ok = row.oi >= cfg["options"]["min_open_interest"]
    if row.bid <= 0 or row.ask <= 0:
        # No live quote (weekend / after hours snapshot): judge on open interest only, spread unknown
        return bool(oi_ok), float("nan"), int(row.oi)
    spread_pct = (row.ask - row.bid) / row.mid
    ok = oi_ok and (spread_pct <= cfg["options"]["max_bid_ask_pct"])
    return bool(ok), float(spread_pct), int(row.oi)


# --------------------------------------------------------------------------- feature bundle
@dataclass
class Features:
    symbol: str
    name: str
    sector: str
    tech: Technicals
    atm_iv: float
    iv_hv20: float
    iv_hv60: float
    iv_rank: float
    iv_pct: float
    iv_method: str
    dte: int
    expiry: str
    next_earnings: date | None
    days_to_earnings: int | None
    earnings_in_window: bool     # earnings on/before the main chain's expiry (should be rare after pre-earnings expiry selection)
    implied_move: float          # ATM straddle / spot on the EVENT chain when present, else the main chain
    event_dte: int | None
    event_expiry: str | None
    event_put_liq: tuple
    event_call_liq: tuple
    realised_moves: list[float]
    info: dict
    put30: pd.Series | None
    call30: pd.Series | None
    put_liq: tuple
    call_liq: tuple
    chain: ChainSnapshot | None = None

    def to_dict(self) -> dict:
        skip = ("put30", "call30", "tech", "chain", "info")
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in skip}
        d.update({f"t_{k}": v for k, v in asdict(self.tech).items()})
        return d


def build_features(sd: SymbolData, uni_row: pd.Series, store: pd.DataFrame, cfg: dict, today: date) -> Features | None:
    main_ok = sd.chain is not None and not sd.chain.calls.empty and not sd.chain.puts.empty
    if not main_ok:
        if sd.event_chain is None or sd.event_chain.calls.empty:
            return None
        sd.chain = sd.event_chain          # earnings-only name: no pre-earnings expiry available
    tech = technicals(sd.prices)
    o = cfg["options"]
    add_deltas(sd.chain, tech.price, o["risk_free_rate"])
    iv = atm_iv(sd.chain, tech.price)
    rank, pct, method = iv_rank_percentile(store, sd.symbol, iv, tech, cfg)
    if method == "proxy":
        rank, pct = proxy_iv_rank(sd.prices, iv)
    put30 = pick_by_delta(sd.chain.puts, o["short_put_delta"])
    call30 = pick_by_delta(sd.chain.calls, o["short_call_delta"])
    dte_e = (sd.next_earnings - today).days if sd.next_earnings else None
    # Earnings on or before expiry disqualify premium-selling strategies (the event is inside the trade)
    in_window = dte_e is not None and 0 <= dte_e <= sd.chain.dte
    ev = sd.event_chain
    ev_ok = ev is not None and not ev.calls.empty and not ev.puts.empty
    ev_put = ev_call = None
    if ev_ok:
        add_deltas(ev, tech.price, o["risk_free_rate"])
        ev_put, ev_call = pick_by_delta(ev.puts, 0.30), pick_by_delta(ev.calls, 0.30)
    implied = straddle_implied_move(ev, tech.price) if ev_ok else straddle_implied_move(sd.chain, tech.price)
    return Features(
        symbol=sd.symbol, name=str(uni_row.get("name", sd.symbol)), sector=str(uni_row.get("sector", "") or sd.info.get("sector", "")),
        tech=tech, atm_iv=iv, iv_hv20=iv / tech.hv20 if tech.hv20 else float("nan"),
        iv_hv60=iv / tech.hv60 if tech.hv60 else float("nan"), iv_rank=rank, iv_pct=pct, iv_method=method,
        dte=sd.chain.dte, expiry=sd.chain.expiry.isoformat(), next_earnings=sd.next_earnings, days_to_earnings=dte_e,
        earnings_in_window=in_window, implied_move=implied,
        event_dte=ev.dte if ev_ok else None, event_expiry=ev.expiry.isoformat() if ev_ok else None,
        event_put_liq=liquidity_ok(ev_put, cfg) if ev_ok else (False, float("nan"), 0),
        event_call_liq=liquidity_ok(ev_call, cfg) if ev_ok else (False, float("nan"), 0),
        realised_moves=realised_earnings_moves(sd.prices, sd.past_earnings), info=sd.info,
        put30=put30, call30=call30, put_liq=liquidity_ok(put30, cfg), call_liq=liquidity_ok(call30, cfg),
        chain=sd.chain,
    )
