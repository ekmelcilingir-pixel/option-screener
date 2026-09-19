"""Market data access (yfinance) plus a deterministic demo generator for offline runs.

Stage 1: batch daily prices for the whole universe (one call).
Stage 2: per-symbol option chain (target expiry), fundamentals, earnings dates.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class ChainSnapshot:
    expiry: date
    dte: int
    calls: pd.DataFrame  # strike, bid, ask, iv, oi, volume
    puts: pd.DataFrame


@dataclass
class SymbolData:
    symbol: str
    prices: pd.DataFrame            # index=date, cols: open high low close volume
    chain: ChainSnapshot | None = None          # main chain: target DTE band, expiring BEFORE earnings when possible
    event_chain: ChainSnapshot | None = None    # first expiry AFTER an earnings date inside the earnings window
    info: dict = field(default_factory=dict)
    next_earnings: date | None = None
    past_earnings: list[date] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- prices
def fetch_prices(symbols: list[str], period: str = "2y") -> dict[str, pd.DataFrame]:
    """Batch download with retries; Yahoo rate-limits bursts, so batches are small and backed off."""
    import yfinance as yf

    def _extract(raw: pd.DataFrame, batch: list[str]) -> dict[str, pd.DataFrame]:
        got = {}
        for s in batch:
            try:
                df = raw[s] if len(batch) > 1 else raw
                df = df.dropna(subset=["Close"]).rename(columns=str.lower)
                if len(df) > 120:
                    got[s] = df[["open", "high", "low", "close", "volume"]]
            except Exception:  # noqa: BLE001
                pass
        return got

    out: dict[str, pd.DataFrame] = {}
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        got: dict[str, pd.DataFrame] = {}
        for attempt in range(4):
            todo = [s for s in batch if s not in got]
            if not todo:
                break
            try:
                raw = yf.download(todo, period=period, auto_adjust=True, group_by="ticker", threads=False, progress=False)
                got.update(_extract(raw, todo))
            except Exception as e:  # noqa: BLE001
                log.warning("prices: batch %d attempt %d failed (%s)", i // batch_size, attempt + 1, e)
            missing = [s for s in batch if s not in got]
            if missing and attempt < 3:
                wait = 5 * (attempt + 1)
                log.info("prices: batch %d missing %d after attempt %d — retry in %ds", i // batch_size, len(missing), attempt + 1, wait)
                time.sleep(wait)
        out.update(got)
        log.info("prices: %d/%d loaded so far", len(out), min(i + batch_size, len(symbols)))
        time.sleep(1.5)
    log.info("prices: %d/%d symbols loaded", len(out), len(symbols))
    return out


# --------------------------------------------------------------------------- per-symbol
def _pick_expiry(expiries: list[str], dte_min: int, dte_max: int, today: date) -> tuple[str, int] | None:
    cands = []
    for e in expiries:
        d = datetime.strptime(e, "%Y-%m-%d").date()
        dte = (d - today).days
        if dte_min <= dte <= dte_max:
            cands.append((abs(dte - (dte_min + dte_max) / 2), e, dte))
    if not cands:
        return None
    _, e, dte = min(cands)
    return e, dte


def _clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame(
        {
            "strike": df["strike"].astype(float),
            "bid": df["bid"].fillna(0).astype(float),
            "ask": df["ask"].fillna(0).astype(float),
            "iv": df["impliedVolatility"].astype(float),
            "oi": df["openInterest"].fillna(0).astype(int),
            "volume": df["volume"].fillna(0).astype(int),
            "last": df["lastPrice"].fillna(0).astype(float) if "lastPrice" in df else 0.0,
        }
    )
    c["mid"] = (c.bid + c.ask) / 2
    # Outside market hours Yahoo often returns 0/0 or very wide quotes; fall back to last trade for mid.
    no_quote = (c.bid <= 0) | (c.ask <= 0)
    c.loc[no_quote & (c["last"] > 0), "mid"] = c.loc[no_quote & (c["last"] > 0), "last"]
    # Yahoo stamps stale contracts (weekends, illiquid strikes) with iv≈1e-5. Keep every priced row;
    # metrics.add_deltas back-solves IV from the mid where Yahoo's figure is missing.
    c.loc[c.iv <= 0.01, "iv"] = float("nan")
    return c[c.mid > 0].sort_values("strike").reset_index(drop=True)


def fetch_symbol_details(sd: SymbolData, cfg: dict, today: date) -> SymbolData:
    """Populate earnings/chain/info on an existing SymbolData (prices already loaded).

    Earnings dates are fetched first: if a report falls inside the earnings window the chain
    is taken from the first expiry AFTER the event (what an earnings play needs); otherwise
    the expiry closest to the target DTE band is used.
    """
    import yfinance as yf

    t = yf.Ticker(sd.symbol)
    o = cfg["options"]
    try:
        ed = t.get_earnings_dates(limit=16)
        if ed is not None and len(ed):
            dates = sorted({d.date() for d in ed.index.to_pydatetime()})
            sd.past_earnings = [d for d in dates if d < today][-cfg["earnings"]["lookback_events"]:]
            fut = [d for d in dates if d >= today]
            sd.next_earnings = fut[0] if fut else None
    except Exception as ex:  # noqa: BLE001
        sd.errors.append(f"earnings: {ex}")

    def _load(e: str) -> ChainSnapshot:
        oc = t.option_chain(e)
        d = datetime.strptime(e, "%Y-%m-%d").date()
        return ChainSnapshot(d, (d - today).days, _clean_chain(oc.calls), _clean_chain(oc.puts))

    try:
        expiries = list(t.options)
        exp_dates = {e: datetime.strptime(e, "%Y-%m-%d").date() for e in expiries}
        pick = _pick_expiry(expiries, o["target_dte_min"], o["target_dte_max"], today)
        dte_e = (sd.next_earnings - today).days if sd.next_earnings else None
        # Premium-selling strategies must not hold through earnings: if the report lands inside the
        # target band, step back to the latest expiry strictly BEFORE the event (>= min_pre_earnings_dte).
        if pick and dte_e is not None and dte_e <= pick[1]:
            before = [e for e in expiries if exp_dates[e] < sd.next_earnings and (exp_dates[e] - today).days >= o["min_pre_earnings_dte"]]
            pick = (before[-1], (exp_dates[before[-1]] - today).days) if before else None
        if pick:
            sd.chain = _load(pick[0])
        # Earnings play: first expiry after the event, only when the event is inside the window
        if dte_e is not None and 0 <= dte_e <= cfg["earnings"]["window_days"]:
            after = [e for e in expiries if exp_dates[e] > sd.next_earnings]
            if after:
                sd.event_chain = _load(after[0])
    except Exception as ex:  # noqa: BLE001
        sd.errors.append(f"chain: {ex}")

    try:
        info = t.info or {}
        keys = ("marketCap", "forwardPE", "trailingPE", "profitMargins", "dividendYield", "sector", "shortName",
                "debtToEquity", "returnOnEquity", "freeCashflow", "beta")
        sd.info = {k: info.get(k) for k in keys}
    except Exception as ex:  # noqa: BLE001
        sd.errors.append(f"info: {ex}")
    return sd


# --------------------------------------------------------------------------- demo data
_DEMO = [
    ("AAPL", 230, 0.24, "Technology"), ("MSFT", 420, 0.22, "Technology"), ("NVDA", 125, 0.48, "Technology"),
    ("AMD", 150, 0.52, "Technology"), ("JPM", 210, 0.20, "Financials"), ("XOM", 115, 0.23, "Energy"),
    ("KO", 68, 0.14, "Consumer Staples"), ("TSLA", 240, 0.60, "Consumer Discretionary"),
    ("META", 520, 0.34, "Communication"), ("PFE", 28, 0.27, "Health Care"), ("AVGO", 170, 0.40, "Technology"),
    ("COST", 880, 0.19, "Consumer Staples"), ("NFLX", 680, 0.36, "Communication"), ("INTC", 22, 0.55, "Technology"),
    ("CRM", 260, 0.31, "Technology"),
]


def _bs_price(kind: str, s: float, k: float, t: float, r: float, iv: float) -> float:
    if t <= 0 or iv <= 0:
        return max(0.0, (s - k) if kind == "call" else (k - s))
    d1 = (math.log(s / k) + (r + iv * iv / 2) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    if kind == "call":
        return s * N(d1) - k * math.exp(-r * t) * N(d2)
    return k * math.exp(-r * t) * N(-d2) - s * N(-d1)


def demo_data(cfg: dict, today: date) -> tuple[pd.DataFrame, dict[str, SymbolData]]:
    """Synthetic but internally consistent data so the whole pipeline can be exercised offline."""
    rng = np.random.default_rng(42)
    uni = pd.DataFrame([{"symbol": s, "name": s, "sector": sec, "source": "demo"} for s, _, _, sec in _DEMO])
    out: dict[str, SymbolData] = {}
    idx = pd.bdate_range(end=today, periods=400)
    for j, (s, px, vol, sec) in enumerate(_DEMO):
        drift = rng.normal(0.0004, 0.0003)
        rets = rng.normal(drift, vol / math.sqrt(252), len(idx))
        close = px * np.exp(np.cumsum(rets))
        close *= px / close[-1]
        prices = pd.DataFrame({"close": close}, index=idx)
        prices["open"] = prices.close.shift(1).fillna(prices.close)
        prices["high"] = prices[["open", "close"]].max(axis=1) * (1 + rng.uniform(0, 0.01, len(idx)))
        prices["low"] = prices[["open", "close"]].min(axis=1) * (1 - rng.uniform(0, 0.01, len(idx)))
        prices["volume"] = rng.integers(2_000_000, 40_000_000, len(idx))
        sd = SymbolData(s, prices)

        # Some names report within the earnings window, some later
        sd.next_earnings = today + timedelta(days=int(rng.choice([5, 9, 12, 30, 45, 60])))
        sd.past_earnings = [today - timedelta(days=91 * n + 3) for n in range(1, 9)][::-1]
        dte = 35
        expiry = today + timedelta(days=dte)
        iv_atm = vol * rng.uniform(0.85, 1.35)
        if (sd.next_earnings - today).days <= 14:
            iv_atm *= rng.uniform(1.05, 1.7)   # event premium
        strikes = np.round(np.arange(px * 0.7, px * 1.3, max(1, round(px * 0.025))), 0)
        rows_c, rows_p = [], []
        for k in strikes:
            m = math.log(k / px)
            iv = iv_atm * (1 + 0.6 * max(0, -m) + 0.2 * max(0, m))  # skew
            for kind, rows in (("call", rows_c), ("put", rows_p)):
                mid = _bs_price(kind, px, k, dte / 365, 0.04, iv)
                spr = max(0.02, mid * rng.uniform(0.02, 0.10))
                oi = int(max(0, rng.normal(3000, 1500) * math.exp(-8 * m * m)))
                rows.append({"strike": k, "bid": max(0, mid - spr / 2), "ask": mid + spr / 2, "iv": iv, "oi": oi,
                             "volume": int(oi * rng.uniform(0.05, 0.4)), "mid": mid})
        sd.chain = ChainSnapshot(expiry, dte, pd.DataFrame(rows_c), pd.DataFrame(rows_p))
        if (sd.next_earnings - today).days <= 14:
            sd.event_chain = ChainSnapshot(expiry, dte, pd.DataFrame(rows_c).copy(), pd.DataFrame(rows_p).copy())
        sd.info = {"marketCap": float(rng.uniform(5e10, 2e12)), "forwardPE": float(rng.uniform(12, 45)),
                   "profitMargins": float(rng.uniform(0.02, 0.35)), "dividendYield": float(rng.choice([0, 0.005, 0.02, 0.03])),
                   "sector": sec, "shortName": s, "debtToEquity": float(rng.uniform(10, 150)),
                   "returnOnEquity": float(rng.uniform(0.05, 0.4)), "beta": float(rng.uniform(0.6, 1.8))}
        out[s] = sd
    return uni, out
