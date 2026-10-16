# SOL/USD hardening 2026-10-13

Paper trading only. This release adds opt-in RLR_TRADER_HARDENED=1:
- Whole UTC session VWAP with 5h earlier history to seed EMAs
- Completed 5-minute EMA candles only
- Exact two-closed-1m-bar VWAP reclaim and >EMA20, positive 5m/60m momentum
- Quote freshness, spread-adjusted VWAP cushion, minimum modeled net target R
- Two-bar VWAP failure exit; recent bid participates in stop checks
- Fee-adjusted daily risk state based on real fill prices and modeled fee rates
- Conservative warning: modeled net R is NOT broker CFEE reconciled net PnL

The governing stop remains a software stop. 30s polling improves cadence
but does NOT guarantee market exit prices. No live keys or live orders.
Replay on previous journal data is diagnostic only, NOT proof of net edge.

## Safety and evidence
- Read-only timer checks Alpaca paper FILL and CFEE every 15 minutes.
- Full fee confirmation is false if any crypto fill lacks matching CFEE; do NOT
  trust matched FILL-only P&L as verified net P&L in that case.
- Until matched CFEE is complete, the governor conservatively *models* fees
  at RLR_CRYPTO_FEE_RATE, default 0.0025 per side. This is a model, not
  broker-confirmed accounting. Audit CFEE independently before real money.
- 30-second order-loop cadence is the first safer improvement. It is NOT
  continuous risk monitoring or an exchange-held stop and can gap below stops.
- All runtime and policy enforcement remains Alpaca PAPER ONLY. No live orders.
- Replay result of the two historical SOL/USD signals: both would have been
  rejected by observable gates. Historical sample is tiny; no profitable edge
  has been proven. Keep paper-only until separate blind validation.
