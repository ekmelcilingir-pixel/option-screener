"""CLI entry point.

  python -m screener.run                 # full universe (S&P 500 + NDX 100)
  python -m screener.run --limit 40      # first N symbols (smoke test)
  python -m screener.run --symbols AAPL,MSFT,NVDA
  python -m screener.run --demo          # offline synthetic data
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from . import __version__
from .market_data import SymbolData, fetch_prices, fetch_symbol_details, demo_data
from .metrics import build_features, technicals, load_iv_store, append_iv_store
from .strategies import ALL_STRATEGIES, Candidate
from .report import render_report
from .universe import load_universe

log = logging.getLogger("screener")


def prefilter_score(t) -> float:
    """Cheap Stage-1 ranking from prices only: liquidity, tradeable trend, HV in a sellable range."""
    s = 0.0
    s += min(t.avg_dollar_vol / 2e8, 1) * 30
    s += 25 if t.trend != "flat" else 10
    s += 25 * (1 - min(abs(t.hv60 - 0.35) / 0.35, 1))     # HV ~20-50% sweet spot
    s += 20 * (1 - min(abs(t.rsi14 - 50) / 40, 1))
    return s


def run(cfg: dict, args) -> Path:
    today = date.today()
    t0 = time.time()
    store = load_iv_store()

    cache = Path("data/cache") / today.isoformat()
    cache.mkdir(parents=True, exist_ok=True)
    deadline = t0 + args.max_seconds if args.max_seconds else None

    if args.demo:
        uni, data = demo_data(cfg, today)
    else:
        uni = load_universe(cfg, refresh=args.refresh_universe)
        symbols = [s for s in uni.symbol.tolist()]
        if args.symbols:
            symbols = args.symbols.split(",")
        elif args.limit:
            symbols = symbols[: args.limit]
        key = "prices_" + (args.symbols.replace(",", "_") if args.symbols else f"n{len(symbols)}") + ".pkl"
        pfile = cache / key
        if pfile.exists():
            prices = pickle.load(pfile.open("rb"))
            log.info("prices: %d symbols from cache", len(prices))
        else:
            prices = fetch_prices(symbols)
            pickle.dump(prices, pfile.open("wb"))
        if not args.symbols and len(prices) < 0.6 * len(symbols):
            pfile.unlink(missing_ok=True)
            log.error("prices: only %d/%d symbols loaded — Yahoo is rate-limiting; wait a few minutes and rerun", len(prices), len(symbols))
            sys.exit(4)
        data = {s: SymbolData(s, p) for s, p in prices.items()}

        # ---- Stage 1: price-only prefilter
        u = cfg["universe"]
        ranked = []
        for s, sd in data.items():
            t = technicals(sd.prices)
            if t.price < u["min_price"] or t.avg_dollar_vol < u["min_avg_dollar_volume"]:
                continue
            ranked.append((prefilter_score(t), s, t))
        ranked.sort(reverse=True)
        keep = [s for _, s, _ in ranked[: cfg["prefilter"]["top_n"]]] if not args.symbols else list(data)
        log.info("stage1: %d passed filters, %d proceed to stage 2", len(ranked), len(keep))

        # ---- Stage 2: chains + fundamentals + earnings (per-symbol cache -> resumable)
        pending = [s for s in keep if not (cache / f"{s}.pkl").exists()]
        log.info("stage2: %d cached, %d to fetch", len(keep) - len(pending), len(pending))
        for i, s in enumerate(pending, 1):
            if deadline and time.time() > deadline:
                log.warning("stage2: time budget reached after %d symbols — rerun to resume (%d left)", i - 1, len(pending) - i + 1)
                sys.exit(3)
            sd = fetch_symbol_details(data[s], cfg, today)
            pickle.dump({"chain": sd.chain, "event_chain": sd.event_chain, "info": sd.info, "next_earnings": sd.next_earnings,
                         "past_earnings": sd.past_earnings, "errors": sd.errors}, (cache / f"{s}.pkl").open("wb"))
            if i % 25 == 0:
                log.info("stage2: %d/%d", i, len(pending))
            time.sleep(0.4)
        for s in keep:
            d = pickle.load((cache / f"{s}.pkl").open("rb"))
            sd = data[s]
            sd.chain, sd.info, sd.next_earnings, sd.past_earnings, sd.errors = d["chain"], d["info"], d["next_earnings"], d["past_earnings"], d["errors"]
            sd.event_chain = d.get("event_chain")
        data = {s: data[s] for s in keep}

    # ---- Features + IV store
    uni_idx = uni.set_index("symbol")
    feats, iv_rows = [], []
    for s, sd in data.items():
        row = uni_idx.loc[s] if s in uni_idx.index else pd.Series({"name": s, "sector": ""})
        try:
            f = build_features(sd, row, store, cfg, today)
        except Exception as e:  # noqa: BLE001
            log.warning("features: %s failed (%s)", s, e)
            continue
        if f is None:
            continue
        feats.append(f)
        iv_rows.append({"symbol": s, "atm_iv": round(f.atm_iv, 4), "hv20": round(f.tech.hv20, 4),
                        "hv60": round(f.tech.hv60, 4), "price": round(f.tech.price, 2)})
    if iv_rows and not args.demo:
        append_iv_store(iv_rows, today)

    # ---- Strategies
    results: dict[str, list[Candidate]] = {}
    rejections: dict[str, dict[str, int]] = {}
    for strat in ALL_STRATEGIES:
        cands, rej = [], {}
        for f in feats:
            try:
                c = strat.evaluate(f, cfg)
            except Exception as e:  # noqa: BLE001
                log.warning("%s: %s failed (%s)", strat.key, f.symbol, e)
                c = f"error: {type(e).__name__}"
            if isinstance(c, Candidate):
                cands.append(c)
            else:
                rej[c or "other"] = rej.get(c or "other", 0) + 1
        cands.sort(key=lambda c: c.score, reverse=True)
        results[strat.key] = cands
        rejections[strat.key] = dict(sorted(rej.items(), key=lambda kv: -kv[1]))
        log.info("%s: %d candidates; rejected %s", strat.key, len(cands), rejections[strat.key])

    # ---- Outputs
    out_dir = Path(cfg["report"]["output_dir"])
    out_dir.mkdir(exist_ok=True)
    meta = {
        "date": today.isoformat(), "version": __version__, "mode": "demo" if args.demo else "live",
        "universe_size": int(len(uni)), "priced": int(len(data)), "featured": int(len(feats)),
        "iv_history_days": int(store.date.nunique()) if len(store) else 0,
        "elapsed_s": round(time.time() - t0, 1),
        "errors": {s: sd.errors for s, sd in data.items() if sd.errors},
        "rejections": rejections,
        "no_chain": [s for s, sd in data.items() if sd.chain is None and sd.event_chain is None],
    }
    html = render_report(results, feats, meta, cfg)
    stamped = out_dir / f"screener_{today.isoformat()}.html"
    stamped.write_text(html, encoding="utf-8")
    (out_dir / "index.html").write_text(html, encoding="utf-8")

    # machine-readable outputs
    rows = []
    for k, cands in results.items():
        for rank, c in enumerate(cands, 1):
            rows.append({"strategy": k, "rank": rank, "symbol": c.symbol, "score": c.score, **{f"setup_{a}": b for a, b in c.setup.items()},
                         **{f"pts_{a}": round(b, 1) for a, b in c.factors.items()}, "notes": "; ".join(c.notes)})
    pd.DataFrame(rows).to_csv(out_dir / f"candidates_{today.isoformat()}.csv", index=False)
    pd.DataFrame([f.to_dict() for f in feats]).to_csv(out_dir / f"features_{today.isoformat()}.csv", index=False)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("report written: %s (%.1fs)", stamped, meta["elapsed_s"])
    return stamped


def _configure_yf_cache() -> None:
    """Keep yfinance's sqlite caches on a local disk (mounted/network FS can raise 'database is locked')."""
    import os, tempfile
    d = os.environ.get("YF_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "yf-cache")
    os.makedirs(d, exist_ok=True)
    try:
        import yfinance as yf
        yf.set_tz_cache_location(d)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    _configure_yf_cache()
    ap = argparse.ArgumentParser(description="OptionScreener")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--max-seconds", type=int, default=0, help="stop stage 2 after N seconds (exit 3); rerun to resume")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    run(cfg, args)


if __name__ == "__main__":
    main()
