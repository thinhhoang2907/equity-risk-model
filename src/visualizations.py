# src/visualizations.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import warnings
import os

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
FIGURES_DIR   = "data/figures"

def _save_fig(fig, filename):
    """
    Saves figure to disk only if the figures directory exists and is writable.
    Silently skips on Streamlit Cloud where the filesystem is read-only.
    """
    try:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.savefig(f"{FIGURES_DIR}/{filename}", dpi=150, bbox_inches="tight")
        print(f"  ✓ {filename}")
    except (OSError, PermissionError):
        pass  # Running on Streamlit Cloud — skip saving, just display

def load_all_outputs():
    """Loads all CSVs produced by Phases 2 and 3."""
    master = pd.read_csv(
        f"{PROCESSED_DIR}/master_dataset.csv",
        index_col="Date", parse_dates=True
    )
    loadings    = pd.read_csv(f"{PROCESSED_DIR}/factor_loadings.csv",    index_col="Ticker")
    decomp      = pd.read_csv(f"{PROCESSED_DIR}/risk_decomposition.csv", index_col="Ticker")
    var_cvar    = pd.read_csv(f"{PROCESSED_DIR}/var_cvar.csv")
    stress      = pd.read_csv(f"{PROCESSED_DIR}/stress_tests.csv")
    rolling     = pd.read_csv(f"{PROCESSED_DIR}/rolling_betas.csv", index_col=0, parse_dates=True)
    rolling.index.name = "Date"

    stock_cols     = [c for c in master.columns if c.endswith("_excess")]
    excess_returns = master[stock_cols].copy()
    excess_returns.columns = [c.replace("_excess", "") for c in stock_cols]

    return master, loadings, decomp, var_cvar, stress, rolling, excess_returns


# ── Chart 1: Factor Loading Heatmap ──────────────────────────────────────────

def plot_factor_heatmap(loadings):
    factor_cols = ["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_MOM"]
    labels      = ["β MKT", "β SMB", "β HML", "β MOM"]
    data        = loadings[factor_cols].copy()
    data.columns = labels

    fig, ax = plt.subplots(figsize=(9, 7))
    vmax    = max(abs(data.values.min()), abs(data.values.max()))
    im      = ax.imshow(data.values, cmap="RdYlGn", aspect="auto",
                        vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=10)

    # Annotate each cell with the value
    for i in range(len(data.index)):
        for j in range(len(labels)):
            val   = data.values[i, j]
            color = "white" if abs(val) > 0.6 * vmax else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, shrink=0.8, label="Factor Loading")
    ax.set_title("Factor Loading Heatmap — FF4 Model", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Fama-French Factor", fontsize=11)
    ax.set_ylabel("Stock", fontsize=11)

    plt.tight_layout()
    _save_fig(fig, "factor_heatmap.png")
    return fig


# ── Chart 2: Risk Decomposition Bar Chart ────────────────────────────────────

def plot_risk_decomposition(decomp):
    tickers = decomp.index.tolist()
    sys_pct  = decomp["Sys_Pct"].values
    idio_pct = decomp["Idio_Pct"].values

    # Sort by systematic % descending
    order    = np.argsort(sys_pct)[::-1]
    tickers  = [tickers[i] for i in order]
    sys_pct  = sys_pct[order]
    idio_pct = idio_pct[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    x       = np.arange(len(tickers))
    width   = 0.6

    bars1 = ax.bar(x, sys_pct,  width, label="Systematic",    color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, idio_pct, width, bottom=sys_pct,
                   label="Idiosyncratic", color="#FF9800", alpha=0.85)

    # Label each segment
    for bar, val in zip(bars1, sys_pct):
        if val > 8:
            ax.text(bar.get_x() + bar.get_width()/2, val/2,
                    f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
    for bar, s, val in zip(bars2, sys_pct, idio_pct):
        if val > 8:
            ax.text(bar.get_x() + bar.get_width()/2, s + val/2,
                    f"{val:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=10)
    ax.set_ylabel("Share of Total Variance (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title("Risk Decomposition: Systematic vs Idiosyncratic",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    _save_fig(fig, "risk_decomposition.png")
    return fig


# ── Chart 3: Return Distribution with VaR/CVaR ───────────────────────────────

def plot_var_distribution(excess_returns, var_cvar):
    tickers      = excess_returns.columns.tolist()
    weights      = np.ones(len(tickers)) / len(tickers)
    port_returns = excess_returns.dot(weights)

    # Pull VaR/CVaR values from the saved CSV
    hist_var95  = var_cvar.loc[var_cvar["Confidence"] == 0.95, "Hist_VaR"].values[0]
    hist_cvar95 = var_cvar.loc[var_cvar["Confidence"] == 0.95, "Hist_CVaR"].values[0]
    param_var95 = var_cvar.loc[var_cvar["Confidence"] == 0.95, "Param_VaR"].values[0]
    hist_var99  = var_cvar.loc[var_cvar["Confidence"] == 0.99, "Hist_VaR"].values[0]

    fig, ax = plt.subplots(figsize=(11, 6))
    n, bins, patches = ax.hist(port_returns, bins=80, color="#1565C0",
                                alpha=0.65, edgecolor="none", density=True)

    # Color the tail red
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < -hist_var95:
            patch.set_facecolor("#D32F2F")
            patch.set_alpha(0.8)

    # Vertical lines
    ax.axvline(-hist_var95,  color="#D32F2F",  linewidth=2,   linestyle="--",
               label=f"Hist VaR 95%  = {hist_var95:.4f}")
    ax.axvline(-hist_cvar95, color="#B71C1C",  linewidth=2,   linestyle="-",
               label=f"Hist CVaR 95% = {hist_cvar95:.4f}")
    ax.axvline(-param_var95, color="#FF6F00",  linewidth=1.5, linestyle=":",
               label=f"Param VaR 95% = {param_var95:.4f}")
    ax.axvline(-hist_var99,  color="#4A148C",  linewidth=1.5, linestyle="--",
               label=f"Hist VaR 99%  = {hist_var99:.4f}")

    ax.set_xlabel("Daily Portfolio Return", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Portfolio Return Distribution with VaR / CVaR Thresholds",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=9, loc="upper left")

    # Annotate fat-tail premium
    ax.text(0.98, 0.95,
            f"Fat-tail premium: +21.1%\n(Hist CVaR vs Param CVaR)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#B71C1C",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#B71C1C", alpha=0.8))

    plt.tight_layout()
    _save_fig(fig, "var_distribution.png")
    return fig


# ── Chart 4: Rolling Beta ─────────────────────────────────────────────────────

def plot_rolling_betas(rolling, highlight=["NVDA", "JNJ", "AAPL", "BAC"]):
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"NVDA": "#F44336", "JNJ": "#4CAF50",
              "AAPL": "#2196F3", "BAC": "#FF9800"}

    for ticker in highlight:
        if ticker in rolling.columns:
            ax.plot(rolling.index, rolling[ticker],
                    label=ticker, linewidth=1.8,
                    color=colors.get(ticker, "gray"), alpha=0.9)

    # Shade crisis periods
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30"),
               alpha=0.12, color="red", label="COVID Crash")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"),
               alpha=0.10, color="orange", label="2022 Rate Hikes")

    ax.axhline(y=1.0, color="gray", linestyle="--",
               linewidth=1, alpha=0.6, label="β = 1.0 (market)")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Rolling 126-day Market Beta", fontsize=11)
    ax.set_title("Rolling Market Beta (6-Month Window) — Selected Stocks",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=9, ncol=3)
    ax.set_ylim(-0.5, 3.5)

    plt.tight_layout()
    _save_fig(fig, "rolling_betas.png")
    return fig


# ── Chart 5: Stress Test Comparison ──────────────────────────────────────────

def plot_stress_tests(stress):
    # Exclude full period from bars — use it as reference line only
    plot_df  = stress[stress["Period"] != "Full Period"].copy()
    full_var = stress.loc[stress["Period"] == "Full Period", "Hist_VaR95"].values[0]
    full_vol = stress.loc[stress["Period"] == "Full Period", "Ann_Vol"].values[0]

    periods  = plot_df["Period"].tolist()
    x        = np.arange(len(periods))
    width    = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # Left: Annualized Vol comparison
    ax = axes[0]
    bars = ax.bar(x, plot_df["Ann_Vol"], width*1.8,
                  color=["#D32F2F", "#1565C0"], alpha=0.8)
    ax.axhline(full_vol, color="gray", linestyle="--",
               linewidth=1.5, label=f"Full period ({full_vol:.4f})")
    for bar, val in zip(bars, plot_df["Ann_Vol"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=9)
    ax.set_ylabel("Annualized Volatility", fontsize=11)
    ax.set_title("Volatility by Stress Period", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(plot_df["Ann_Vol"].max(), full_vol) * 1.25)

    # Right: VaR95 comparison
    ax = axes[1]
    bars = ax.bar(x, plot_df["Hist_VaR95"], width*1.8,
                  color=["#D32F2F", "#1565C0"], alpha=0.8)
    ax.axhline(full_var, color="gray", linestyle="--",
               linewidth=1.5, label=f"Full period ({full_var:.5f})")
    for bar, val in zip(bars, plot_df["Hist_VaR95"]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f"{val:.5f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=9)
    ax.set_ylabel("Daily VaR (95%)", fontsize=11)
    ax.set_title("VaR95 by Stress Period", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(plot_df["Hist_VaR95"].max(), full_var) * 1.25)

    plt.suptitle("Stress Test Comparison vs Full Period Baseline",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, "stress_tests.png")
    return fig


# ── Chart 6: Efficient Frontier ───────────────────────────────────────────────

def plot_efficient_frontier(excess_returns, n_portfolios=4000):
    """
    Simulates random portfolios by drawing random weight vectors,
    computes each portfolio's annualized return and volatility,
    and colors them by Sharpe ratio.
    Also marks the equal-weight and max-Sharpe portfolios.
    """
    tickers     = excess_returns.columns.tolist()
    n           = len(tickers)
    mean_daily  = excess_returns.mean().values
    cov_daily   = excess_returns.cov().values

    port_vols    = []
    port_rets    = []
    port_sharpes = []
    port_weights = []

    np.random.seed(42)
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n))  # random weights summing to 1
        r = np.dot(w, mean_daily) * 252
        v = np.sqrt(w @ cov_daily @ w) * np.sqrt(252)
        s = r / v
        port_vols.append(v)
        port_rets.append(r)
        port_sharpes.append(s)
        port_weights.append(w)

    port_vols    = np.array(port_vols)
    port_rets    = np.array(port_rets)
    port_sharpes = np.array(port_sharpes)

    # Max Sharpe portfolio
    max_idx    = np.argmax(port_sharpes)
    max_weights = port_weights[max_idx]

    # Equal weight portfolio
    ew          = np.ones(n) / n
    ew_ret      = np.dot(ew, mean_daily) * 252
    ew_vol      = np.sqrt(ew @ cov_daily @ ew) * np.sqrt(252)
    ew_sharpe   = ew_ret / ew_vol

    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(port_vols, port_rets, c=port_sharpes,
                    cmap="viridis", alpha=0.4, s=8)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # Mark max Sharpe
    ax.scatter(port_vols[max_idx], port_rets[max_idx],
               color="red", s=180, zorder=5, marker="*",
               label=f"Max Sharpe ({port_sharpes[max_idx]:.2f})")

    # Mark equal weight
    ax.scatter(ew_vol, ew_ret, color="white", s=120, zorder=5,
               marker="D", edgecolors="black", linewidths=1.5,
               label=f"Equal Weight (SR={ew_sharpe:.2f})")

    # Label individual stocks
    for ticker in tickers:
        r = excess_returns[ticker].mean() * 252
        v = excess_returns[ticker].std() * np.sqrt(252)
        ax.annotate(ticker, (v, r), textcoords="offset points",
                    xytext=(5, 3), fontsize=7.5, color="black",
                    fontweight="bold")
        ax.scatter(v, r, color="black", s=30, zorder=4, alpha=0.7)

    ax.set_xlabel("Annualized Volatility", fontsize=11)
    ax.set_ylabel("Annualized Return", fontsize=11)
    ax.set_title("Efficient Frontier — 4,000 Random Portfolios",
                 fontsize=13, fontweight="bold", pad=15)
    ax.legend(fontsize=9)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.tick_params(colors="black")

    plt.tight_layout()
    _save_fig(fig, "efficient_frontier.png")

    # Print max Sharpe weights for reference
    print("\n  Max Sharpe portfolio weights:")
    for ticker, w in zip(tickers, max_weights):
        print(f"    {ticker:<6} {w:.3f}")
    print(f"  Max Sharpe ratio: {port_sharpes[max_idx]:.3f}")
    print(f"  Equal-weight Sharpe ratio: {ew_sharpe:.3f}")

    return fig


# ── Main runner ───────────────────────────────────────────────────────────────

def run_visualizations():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    master, loadings, decomp, var_cvar, stress, rolling, excess_returns = load_all_outputs()

    print("Generating charts...")
    fig1 = plot_factor_heatmap(loadings)
    fig2 = plot_risk_decomposition(decomp)
    fig3 = plot_var_distribution(excess_returns, var_cvar)
    fig4 = plot_rolling_betas(rolling)
    fig5 = plot_stress_tests(stress)
    fig6 = plot_efficient_frontier(excess_returns)

    print(f"\nAll charts saved to {FIGURES_DIR}/")
    return fig1, fig2, fig3, fig4, fig5, fig6


if __name__ == "__main__":
    run_visualizations()