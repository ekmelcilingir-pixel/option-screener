# OptionScreener

Daily screener that ranks S&P 500 + Nasdaq 100 names by suitability for four option strategies, using free Yahoo Finance data (yfinance). Runs on GitHub Actions after the US close, publishes a self-contained HTML report to GitHub Pages, and accumulates its own ATM-IV history so IV Rank becomes native after ~60 trading days.

## Strategies

| Key | Strategy | Looks for | Hard filters |
|---|---|---|---|
| `csp` | Cash-secured put / Wheel | High IV rank, IV > HV, uptrend or pullback to support, quality fundamentals, liquid 30Δ put | no earnings before expiry, not in downtrend |
| `cc` | Covered call / Buy-write | IV rich vs realised, low-beta stable name, moderate momentum (RSI 42–62), dividend bonus | no earnings before expiry |
| `spread` | Bull put / Bear call | Persistent trend (SMA50/200 + slope), IV rank, cushion to short strike, return-on-risk 25–45% | trend must be non-flat, no earnings |
| `earnings` | Straddle / Strangle / Iron condor | Straddle-implied *event* move vs realised moves over last 8 reports; ratio ≥1.15 → sell premium, ≤0.85 → buy vol | earnings within 14 days, ≥4 past events, liquid chain |

Every score is 0–100 with the factor breakdown shown in the report and CSV. Weights live in `screener/strategies/*.py`; thresholds in `config.yaml`.

## Pipeline

1. **Universe** — S&P 500 + NDX 100 from Wikipedia, cached in `data/universe.csv` (`--refresh-universe` to rebuild).
2. **Stage 1** — one batch price download; cheap prefilter (liquidity, trend, HV band, RSI). Top 150 continue.
3. **Stage 2** — per symbol: earnings dates → option chain (25–50 DTE, or first expiry after earnings if a report is within 14 days) → fundamentals.
4. **Features** — HV20/60/252, SMA/RSI/trend, ATM IV, IV/HV, IV Rank (stored history or HV-distribution proxy, flagged), Black–Scholes deltas, 30Δ strikes, spread/OI liquidity, straddle implied move, realised earnings moves.
5. **Scoring** — each strategy applies hard filters then weighted factors.
6. **Outputs** — `reports/index.html` (+ dated copy), `candidates_<date>.csv`, `features_<date>.csv`, `meta.json`; `data/iv_history.csv` appended.

## Run

```bash
pip install -r requirements.txt
python -m screener.run --demo                 # offline synthetic data, exercises everything
python -m screener.run --symbols AAPL,MSFT    # quick live check
python -m screener.run --limit 60             # partial universe
python -m screener.run                        # full run (~20–40 min, yfinance rate limits)
```

## Deploy

1. Push to a GitHub repo. Settings → Pages → Source: **GitHub Actions**.
2. Settings → Actions → General → Workflow permissions: **Read and write**.
3. Actions → *Daily OptionScreener* → Run workflow (optionally `limit=40` for a smoke test).

## Extending

- Swap the data layer: `market_data.py` exposes `fetch_prices` and `fetch_symbol_details`; an IBKR/Unusual Whales implementation only needs to return the same `SymbolData`.
- IV Rank: once `data/iv_history.csv` holds ≥60 days per symbol the proxy switches off automatically.
- New strategy: subclass `Strategy` in `screener/strategies/`, register in `ALL_STRATEGIES`, add a column block in `templates/report.html.j2`.

Not investment advice. Yahoo data is delayed and occasionally missing; failures are logged per symbol in `meta.json`.
