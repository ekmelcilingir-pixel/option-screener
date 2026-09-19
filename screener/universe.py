"""Universe construction: S&P 500 + Nasdaq 100 constituents (Wikipedia), cached locally."""
from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

WIKI = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "Symbol", 0),
    "ndx100": ("https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies", "Ticker", None),
}

CACHE = Path("data/universe.csv")


def _fetch_table(url: str, col: str, idx: int | None) -> pd.DataFrame:
    import requests

    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    if idx is not None:
        t = tables[idx]
    else:
        t = next(t for t in tables if col in t.columns)
    out = pd.DataFrame({"symbol": t[col].astype(str).str.replace(".", "-", regex=False).str.strip()})
    cols = {str(c): c for c in t.columns}
    name_col = next((cols[c] for c in cols if c in ("Security", "Company")), None)
    sec_col = next((cols[c] for c in cols if c.startswith("GICS Sector") or c.startswith("ICB Industry")), None)
    out["name"] = t[name_col] if name_col is not None else out.symbol
    out["sector"] = t[sec_col] if sec_col is not None else ""
    return out


def load_universe(cfg: dict, refresh: bool = False) -> pd.DataFrame:
    """Return DataFrame[symbol, name, sector, source]. Uses cache unless refresh=True."""
    if CACHE.exists() and not refresh:
        df = pd.read_csv(CACHE)
    else:
        frames = []
        for src in cfg["universe"]["sources"]:
            url, col, idx = WIKI[src]
            try:
                t = _fetch_table(url, col, idx)
                t["source"] = src
                frames.append(t)
                log.info("universe: %s -> %d symbols", src, len(t))
            except Exception as e:  # noqa: BLE001
                log.warning("universe: failed to fetch %s (%s)", src, e)
        if not frames:
            raise RuntimeError("Could not build universe and no cache present")
        df = pd.concat(frames, ignore_index=True)
        df = (
            df.groupby("symbol", as_index=False)
            .agg(name=("name", "first"), sector=("sector", "first"), source=("source", lambda s: "+".join(sorted(set(s)))))
        )
        CACHE.parent.mkdir(exist_ok=True)
        df.to_csv(CACHE, index=False)

    extra = cfg["universe"].get("extra_symbols") or []
    excl = set(cfg["universe"].get("exclude_symbols") or [])
    for s in extra:
        if s not in df.symbol.values:
            df.loc[len(df)] = {"symbol": s, "name": s, "sector": "", "source": "manual"}
    return df[~df.symbol.isin(excl)].reset_index(drop=True)
