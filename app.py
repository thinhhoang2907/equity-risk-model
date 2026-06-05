# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader    import run_pipeline
from src.factor_model   import run_factor_model
from src.risk_metrics   import run_risk_metrics, compute_portfolio_returns, equal_weights, historical_var_cvar
from src.visualizations import (
    load_all_outputs,
    plot_factor_heatmap,
    plot_risk_decomposition,
    plot_var_distribution,
    plot_rolling_betas,
    plot_stress_tests,
    plot_efficient_frontier,
)

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Equity Risk Factor Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    return load_all_outputs()

master, loadings, decomp, var_cvar, stress, rolling, excess_returns = load_data()
tickers = excess_returns.columns.tolist()

# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("Portfolio Settings")
st.sidebar.markdown("Adjust weights below. Values are normalized to sum to 1.")

weight_inputs = {}
for ticker in tickers:
    weight_inputs[ticker] = st.sidebar.slider(
        ticker, min_value=0.0, max_value=1.0,
        value=round(1 / len(tickers), 3), step=0.01
    )

raw_weights = pd.Series(weight_inputs)
if raw_weights.sum() == 0:
    st.sidebar.error("At least one weight must be > 0")
    st.stop()
weights = raw_weights / raw_weights.sum()  # normalize to sum to 1

st.sidebar.markdown("---")
st.sidebar.markdown("**Effective weights (normalized):**")
for t, w in weights.items():
    st.sidebar.markdown(f"- {t}: `{w:.3f}`")

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Equity Risk Factor Model")
st.markdown(
    "A four-factor Fama-French risk attribution dashboard built on 5 years "
    "of daily returns for 13 S&P 500 stocks. Adjust portfolio weights in the sidebar."
)

# ── Summary metrics row ───────────────────────────────────────────────────────

port_returns = excess_returns[tickers].dot(weights)
port_ann_vol = port_returns.std() * np.sqrt(252)
port_ann_ret = port_returns.mean() * 252
port_sharpe  = port_ann_ret / port_ann_vol
var95, cvar95 = historical_var_cvar(port_returns, 0.95)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Ann. Return",    f"{port_ann_ret:.1%}")
col2.metric("Ann. Volatility", f"{port_ann_vol:.1%}")
col3.metric("Sharpe Ratio",   f"{port_sharpe:.2f}")
col4.metric("VaR 95% (Daily)", f"{var95:.3%}")
col5.metric("CVaR 95% (Daily)", f"{cvar95:.3%}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "Factor Model",
    "Risk Metrics",
    "Stress Tests",
    "Efficient Frontier",
])

# ── Tab 1: Factor Model ───────────────────────────────────────────────────────

with tab1:
    st.subheader("Factor Loading Heatmap")
    st.markdown(
        "Each cell shows a stock's sensitivity (beta) to a Fama-French factor. "
        "Green = positive loading, Red = negative. "
        "Growth stocks (AAPL, NVDA) show strongly negative HML; "
        "value stocks (BAC, JPM) show strongly positive HML."
    )
    fig1 = plot_factor_heatmap(loadings)
    st.pyplot(fig1)
    plt.close(fig1)

    st.markdown("---")
    st.subheader("Factor Loadings Table")
    display_cols = ["Alpha", "Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_MOM", "R2", "Adj_R2"]
    st.dataframe(
        loadings[display_cols].style.background_gradient(
            cmap="RdYlGn", subset=["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_MOM"]
        ).format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Rolling Market Beta (6-Month Window)")
    st.markdown(
        "Shows how each stock's market sensitivity changed over time. "
        "Note the spike in NVDA's beta during 2023–2024 as AI demand amplified its "
        "correlation with broader market risk-on/risk-off moves."
    )
    fig4 = plot_rolling_betas(rolling)
    st.pyplot(fig4)
    plt.close(fig4)

# ── Tab 2: Risk Metrics ───────────────────────────────────────────────────────

with tab2:
    st.subheader("Risk Decomposition: Systematic vs Idiosyncratic")
    st.markdown(
        "Systematic risk is driven by factor exposure (undiversifiable). "
        "Idiosyncratic risk is company-specific (diversifiable). "
        "At the portfolio level, diversification reduces idiosyncratic share to under 8%."
    )
    fig2 = plot_risk_decomposition(decomp)
    st.pyplot(fig2)
    plt.close(fig2)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Portfolio Risk Decomposition")
        port_metrics = {
            "Total Ann. Vol"      : f"{port_ann_vol:.4f}",
            "Systematic Share"    : f"{decomp['Sys_Pct'].mean():.1f}%  (stock avg)",
            "Idiosyncratic Share" : f"{decomp['Idio_Pct'].mean():.1f}% (stock avg)",
        }
        for k, v in port_metrics.items():
            st.metric(k, v)

    with col_b:
        st.subheader("VaR / CVaR Summary")
        st.dataframe(
            var_cvar[["Confidence", "Hist_VaR", "Hist_CVaR",
                       "Param_VaR", "Param_CVaR"]].style.format("{:.5f}",
                       subset=["Hist_VaR", "Hist_CVaR", "Param_VaR", "Param_CVaR"])
                      .format("{:.0%}", subset=["Confidence"]),
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("Return Distribution with VaR / CVaR Thresholds")
    st.markdown(
        "Red bars = tail losses beyond 95% VaR. "
        "The gap between historical and parametric CVaR is the **fat-tail premium** — "
        "21.1% here — representing how much the normality assumption underestimates real tail risk."
    )
    fig3 = plot_var_distribution(excess_returns, var_cvar)
    st.pyplot(fig3)
    plt.close(fig3)

# ── Tab 3: Stress Tests ───────────────────────────────────────────────────────

with tab3:
    st.subheader("Stress Test Comparison")
    st.markdown(
        "Re-runs risk metrics on isolated crisis periods. "
        "COVID VaR was **229% higher** than the full-period baseline — "
        "illustrating why static VaR dangerously underestimates risk during regime shifts."
    )
    fig5 = plot_stress_tests(stress)
    st.pyplot(fig5)
    plt.close(fig5)

    st.markdown("---")
    st.subheader("Stress Test Data")
    st.dataframe(
        stress.style.format({
            "Ann_Vol"      : "{:.4f}",
            "Max_Drawdown" : "{:.4f}",
            "Total_Return" : "{:.4f}",
            "Hist_VaR95"   : "{:.5f}",
            "Hist_CVaR95"  : "{:.5f}",
        }),
        use_container_width=True
    )

# ── Tab 4: Efficient Frontier ──────────────────────────────────────────────────

with tab4:
    st.subheader("Efficient Frontier")
    st.markdown(
        "4,000 randomly sampled portfolios colored by Sharpe ratio. "
        "The ⭐ marks the max-Sharpe portfolio; the ◆ marks your current equal-weight allocation. "
        "Individual stocks are labeled in white."
    )
    fig6 = plot_efficient_frontier(excess_returns)
    st.pyplot(fig6)
    plt.close(fig6)

st.markdown("---")
st.caption(
    "Built with Python · Fama-French 4-Factor Model · "
    "Data: Yahoo Finance + Ken French Data Library · "
    "Thinh Hoang . "
)