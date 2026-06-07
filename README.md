# Equity Risk Factor Model

A quantitative risk attribution dashboard built on the **Fama-French 4-Factor Model**, 
applied to a 13-stock S&P 500 portfolio using 5 years of daily return data (2019–2026).

Built to demonstrate factor-based risk decomposition, tail risk measurement, and 
stress testing — core skills for Quantitative Risk and Data Science roles in finance.

**[Demo](https://equity-risk-model-thinh.streamlit.app/)**


---

## What It Does

This project answers three questions a risk analyst asks about any portfolio:

1. **Where does the risk come from?**  
   Decomposes each stock's variance into systematic (factor-driven) and idiosyncratic 
   (company-specific) components using OLS regression on the FF4 factors.

2. **How bad can losses get?**  
   Computes Value at Risk (VaR) and Conditional VaR (Expected Shortfall) at 95% and 99% 
   confidence using both historical simulation and parametric methods.

3. **How does the portfolio behave under stress?**  
   Re-runs all risk metrics on isolated crisis periods: COVID crash (Feb–Apr 2020) 
   and the 2022 rate hike cycle.

---

## Key Findings

**Factor Exposures**
- Growth stocks (NVDA, AAPL, AMZN, MSFT) show strongly negative HML loadings (–0.32 to –0.90), 
  confirming anti-value orientation consistent with high price-to-book ratios.
- Defensive stocks (JNJ, PG) have market betas of 0.44 and 0.52 vs. NVDA's 1.75 — 
  a 3x difference in market sensitivity within the same portfolio.
- NVDA is the only stock with a statistically significant alpha (annualized +26%, p=0.033), 
  reflecting the AI/GPU demand shock that historical factor structure did not anticipate.

**Risk Decomposition**
- At the stock level, idiosyncratic risk ranges from 23% (BAC) to 77% (JNJ).
- At the portfolio level, diversification reduces idiosyncratic risk to just 7.6% of 
  total variance — the remaining 93.2% is systematic and cannot be diversified away.

**Tail Risk**
- Historical CVaR exceeds parametric CVaR by 21.1% at 95% confidence — the fat-tail 
  premium from assuming normality in daily equity returns.
- At 99% confidence, historical VaR (3.91%) exceeds parametric VaR (3.01%) by 30%, 
  showing the normality assumption breaks down most severely in deep tail scenarios.

**Stress Testing**
- COVID crash: daily VaR95 spiked to 6.12% — 229% above the full-period baseline of 1.86%.
- 2022 rate hikes: annualized vol of 25.4% with a max drawdown of –26.2%, driven by 
  the portfolio's growth-stock concentration getting repriced as rates rose.
- A risk manager relying on full-period VaR in January 2020 would have been exposed to 
  losses more than 3x their model's prediction within weeks.

---

## Methodology

### Factor Model
Each stock's excess return is regressed on four factors:
    R_i - RF = α + β_MKT(MKT-RF) + β_SMB(SMB) + β_HML(HML) + β_MOM(MOM) + ε

- **MKT-RF**: Market excess return — broad market sensitivity
- **SMB**: Small Minus Big — size factor
- **HML**: High Minus Low — value vs. growth orientation  
- **MOM**: Momentum — persistence of recent performance
- **α (Alpha)**: Return unexplained by the four factors

Estimated via OLS using `statsmodels`. R² ranges from 0.23 (JNJ) to 0.77 (BAC).

### Risk Decomposition
Portfolio variance is decomposed as:

Total Variance = w'Σw
Systematic     = w' B Σ_F B' w
Idiosyncratic  = w' D w

Where B is the factor loading matrix, Σ_F is the factor covariance matrix, 
and D is the diagonal residual variance matrix.

Covariance estimated using both sample covariance and Ledoit-Wolf shrinkage 
(shrinkage coefficient: 0.014 — small given T/N ≈ 141).

### VaR and CVaR
- **Historical simulation**: empirical percentiles of the realized return distribution
- **Parametric**: assumes normality — `VaR = -(μ - z·σ)`, `CVaR = -(μ - σ·φ(z)/(1-c))`
- Both computed at 95% and 99% confidence, daily and annualized

---

## Portfolio

| Ticker | Sector | Type |
|--------|--------|------|
| AAPL, MSFT, GOOGL, NVDA, AMZN | Technology | Growth |
| JPM, BAC | Financials | Value |
| JNJ, PG | Healthcare / Consumer Staples | Defensive |
| XOM, CVX | Energy | Cyclical / Value |
| CAT, DE | Industrials | Cyclical |

---

## Project Structure

```
equity-risk-model/
├── src/
│   ├── data_loader.py       # Price + FF factor pipeline
│   ├── factor_model.py      # OLS regressions, rolling betas
│   ├── risk_metrics.py      # VaR, CVaR, decomposition, stress tests
│   └── visualizations.py   # All charts
├── app.py                   # Streamlit dashboard
├── data/
│   ├── processed/           # Model outputs (CSVs)
│   └── figures/             # Static chart exports
└── requirements.txt
```

## Limitations

- **Survivorship bias**: All 13 stocks were selected knowing they are current S&P 500 
  constituents. A rigorous backtest would use point-in-time index membership.
- **No 2008 data**: The dataset starts January 2019. The 2008 financial crisis — the 
  canonical stress event for financials — is not captured. BAC and JPM would likely 
  show much larger tail losses in that regime.
- **Normality assumption**: Parametric VaR assumes normally distributed returns. 
  The Jarque-Bera test formally rejects normality for all 13 stocks (kurtosis 6–9), 
  making parametric CVaR systematically conservative at moderate confidence levels 
  and too optimistic in deep tails.
- **Static factor loadings**: OLS estimates a single beta over the full period. 
  Rolling betas show meaningful time variation, particularly around COVID and the 
  2022 rate cycle. A Kalman filter or DCC-GARCH model would capture this dynamically.
- **Equal-weight baseline**: The dashboard defaults to equal weighting. Real 
  portfolio construction would incorporate transaction costs, liquidity constraints, 
  and turnover limits.

---

## Tech Stack

Python · pandas · numpy · statsmodels · scikit-learn · scipy · 
matplotlib · plotly · Streamlit · yfinance · Ken French Data Library