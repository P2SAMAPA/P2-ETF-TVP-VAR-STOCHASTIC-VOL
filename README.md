# TVP‑VAR with Stochastic Volatility Engine for ETFs

Implements a time‑varying parameter vector autoregression (TVP‑VAR) with stochastic volatility. The model captures changing relationships between ETF returns and macro variables (VIX, DXY, yields). The per‑ETF score is the impulse response to a macro shock over a short horizon – a measure of macro sensitivity and potential alpha.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Endogenous variables: ETF returns + selected macro variables
- Recursive estimation with exponential forgetting (adaptive coefficients)
- Impulse response analysis using companion form
- Score = cumulative impulse response at a fixed horizon (e.g., 5 days) to a one‑σ shock in VIX
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-tvp-var-stochastic-vol-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast, O(n_vars² T))
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- Positive impulse response → ETF tends to rise after a positive macro shock (e.g., VIX increase) under the current regime.
- Negative response → ETF is negatively affected.
- Time‑varying nature captures evolving market structure.

## Requirements

See `requirements.txt`.
