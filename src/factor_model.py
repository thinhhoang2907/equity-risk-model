# src/factor_model.py

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
OUTPUT_DIR    = "data/processed"

# ── Step 1: Load master dataset ───────────────────────────────────────────────

def load_master():
    """
    Loads the master dataset produced by data_loader.py.
    Separates it into:
      - excess_returns: each stock's return minus the risk-free rate
      - factors: the four FF factors (right-hand side variables)
    """
    master = pd.read_csv(
        f"{PROCESSED_DIR}/master_dataset.csv",
        index_col="Date",
        parse_dates=True
    )

    factor_cols = ["MKT_RF", "SMB", "HML", "MOM"]
    stock_cols  = [c for c in master.columns if c.endswith("_excess")]

    excess_returns = master[stock_cols]
    factors        = master[factor_cols]

    print(f"Loaded master dataset: {master.shape[0]} days, {len(stock_cols)} stocks")
    return excess_returns, factors


# ── Step 2: Run OLS regression for one stock ──────────────────────────────────

def run_single_regression(stock_excess, factors, ticker_name):
    """
    Runs OLS regression for one stock:
      R_i - RF = α + β_MKT(MKT-RF) + β_SMB(SMB) + β_HML(HML) + β_MOM(MOM) + ε

    We add a constant (sm.add_constant) to estimate alpha.
    Returns the fitted model object so we can extract everything we need.
    """
    X = sm.add_constant(factors)  # adds intercept column named 'const'
    y = stock_excess

    model  = sm.OLS(y, X, missing="drop").fit()
    return model


# ── Step 3: Run regressions for all stocks ────────────────────────────────────

def run_all_regressions(excess_returns, factors):
    """
    Loops over all stocks, runs OLS for each, and stores:
      1. A summary table of coefficients + stats (one row per stock)
      2. A residuals dataframe (used in Phase 3 for idiosyncratic risk)
      3. The raw model objects (for deeper inspection if needed)
    """
    results     = []
    residuals   = {}
    models      = {}

    stock_cols = excess_returns.columns.tolist()

    print(f"\nRunning OLS regressions for {len(stock_cols)} stocks...")
    print("─" * 60)

    for col in stock_cols:
        ticker = col.replace("_excess", "")
        model  = run_single_regression(excess_returns[col], factors, ticker)

        # ── Extract key statistics ────────────────────────────────────────
        alpha       = model.params["const"]
        beta_mkt    = model.params["MKT_RF"]
        beta_smb    = model.params["SMB"]
        beta_hml    = model.params["HML"]
        beta_mom    = model.params["MOM"]
        r_squared   = model.rsquared
        adj_r2      = model.rsquared_adj

        # t-statistics for each coefficient
        t_alpha     = model.tvalues["const"]
        t_mkt       = model.tvalues["MKT_RF"]
        t_smb       = model.tvalues["SMB"]
        t_hml       = model.tvalues["HML"]
        t_mom       = model.tvalues["MOM"]

        # p-values
        p_alpha     = model.pvalues["const"]
        p_mkt       = model.pvalues["MKT_RF"]

        results.append({
            "Ticker"    : ticker,
            "Alpha"     : round(alpha,    6),
            "Beta_MKT"  : round(beta_mkt, 4),
            "Beta_SMB"  : round(beta_smb, 4),
            "Beta_HML"  : round(beta_hml, 4),
            "Beta_MOM"  : round(beta_mom, 4),
            "R2"        : round(r_squared, 4),
            "Adj_R2"    : round(adj_r2,    4),
            "t_Alpha"   : round(t_alpha,   3),
            "t_MKT"     : round(t_mkt,     3),
            "t_SMB"     : round(t_smb,     3),
            "t_HML"     : round(t_hml,     3),
            "t_MOM"     : round(t_mom,     3),
            "p_Alpha"   : round(p_alpha,   4),
            "p_MKT"     : round(p_mkt,     4),
        })

        # Store residuals under the ticker name
        residuals[ticker] = model.resid
        models[ticker]    = model

        # Print a one-line summary per stock
        sig = "***" if p_alpha < 0.01 else "**" if p_alpha < 0.05 else "*" if p_alpha < 0.10 else ""
        print(
            f"  {ticker:<6} | α={alpha:+.5f}{sig:<3} | "
            f"β_MKT={beta_mkt:.3f} | β_SMB={beta_smb:+.3f} | "
            f"β_HML={beta_hml:+.3f} | β_MOM={beta_mom:+.3f} | "
            f"R²={r_squared:.3f}"
        )

    print("─" * 60)
    print("  Significance: *** p<0.01  ** p<0.05  * p<0.10\n")

    # Build summary dataframe
    results_df  = pd.DataFrame(results).set_index("Ticker")
    residuals_df = pd.DataFrame(residuals)
    residuals_df.index = excess_returns.index

    # Save both
    results_df.to_csv(f"{OUTPUT_DIR}/factor_loadings.csv")
    residuals_df.to_csv(f"{OUTPUT_DIR}/residuals.csv")
    print(f"  → Saved factor_loadings.csv and residuals.csv")

    return results_df, residuals_df, models


# ── Step 4: Sanity checks on regression output ────────────────────────────────

def run_regression_sanity_checks(results_df):
    """
    Validates that the regression results make economic sense.
    These are the checks you'd do before presenting results to a PM or risk committee.
    """
    print("\n── Regression Sanity Checks ────────────────────────────────────")

    # 1. Market betas should cluster near 1.0 for large-cap US stocks
    avg_mkt_beta = results_df["Beta_MKT"].mean()
    print(f"\nAverage market beta (Beta_MKT): {avg_mkt_beta:.3f}")
    print("  Expected: 0.8–1.2 for large-cap S&P 500 stocks")
    if 0.7 <= avg_mkt_beta <= 1.3:
        print("  ✓ Looks reasonable")
    else:
        print("  ✗ Unusually high or low — check your excess return calculation")

    # 2. Defensive stocks should have lower betas than tech/cyclicals
    defensive = ["JNJ", "PG"]
    aggressive = ["NVDA", "AMZN", "AAPL"]
    def_betas  = results_df.loc[defensive, "Beta_MKT"].mean()
    agg_betas  = results_df.loc[aggressive, "Beta_MKT"].mean()
    print(f"\nDefensive avg Beta_MKT (JNJ, PG):          {def_betas:.3f}")
    print(f"Aggressive avg Beta_MKT (NVDA, AMZN, AAPL): {agg_betas:.3f}")
    if def_betas < agg_betas:
        print("  ✓ Defensive stocks have lower market exposure — makes sense")
    else:
        print("  ✗ Unexpected — defensive betas should be lower than tech betas")

    # 3. Growth stocks should have negative HML (they're not value stocks)
    growth = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
    growth_hml = results_df.loc[growth, "Beta_HML"].mean()
    print(f"\nGrowth stock avg Beta_HML: {growth_hml:.3f}")
    print("  Expected: negative (growth stocks are anti-value)")
    if growth_hml < 0:
        print("  ✓ Negative HML loading for growth stocks — correct")
    else:
        print("  ✗ Positive HML for growth stocks is unexpected")

    # 4. R² should generally be above 0.5 for large-cap stocks
    low_r2 = results_df[results_df["R2"] < 0.5]
    if low_r2.empty:
        print(f"\n✓ All stocks have R² > 0.50")
    else:
        print(f"\n  ✗ Low R² stocks (model explains less than 50% of variance):")
        print(low_r2[["R2", "Adj_R2"]].to_string())

    # 5. Annualized alpha — should be close to zero for efficient markets
    ann_alpha = results_df["Alpha"] * 252
    print(f"\nAnnualized alphas (should be close to 0 in efficient markets):")
    print(ann_alpha.round(4).to_string())

    print("\n────────────────────────────────────────────────────────────────\n")


# ── Step 5: Print one full regression summary (for learning / README) ─────────

def print_full_summary(models, ticker="AAPL"):
    """
    Prints the full statsmodels regression table for one stock.
    Good to include a screenshot of this in your README —
    it shows you know how to read and interpret regression output.
    """
    print(f"\n── Full OLS Summary: {ticker} ──────────────────────────────────")
    print(models[ticker].summary())
    print()


# ── Step 6: Rolling beta analysis ────────────────────────────────────────────

def compute_rolling_betas(excess_returns, factors, window=126):
    """
    Computes rolling 126-day (6-month) market betas for all stocks.
    This shows how a stock's market sensitivity changes over time —
    particularly interesting around crises (2020 COVID, 2022 rate hikes).
    window=126 is roughly 6 months of trading days.
    """
    print(f"Computing rolling {window}-day betas...")

    stock_cols   = excess_returns.columns.tolist()
    rolling_betas = {}

    for col in stock_cols:
        ticker  = col.replace("_excess", "")
        betas   = []
        dates   = []

        for i in range(window, len(excess_returns)):
            y_window = excess_returns[col].iloc[i-window:i]
            X_window = sm.add_constant(factors.iloc[i-window:i])
            try:
                roll_model = sm.OLS(y_window, X_window, missing="drop").fit()
                betas.append(roll_model.params["MKT_RF"])
            except Exception:
                betas.append(np.nan)
            dates.append(excess_returns.index[i])

        rolling_betas[ticker] = pd.Series(betas, index=dates)

    rolling_df = pd.DataFrame(rolling_betas)
    rolling_df.to_csv(f"{OUTPUT_DIR}/rolling_betas.csv")
    print(f"  → Rolling betas saved")
    return rolling_df


# ── Main runner ───────────────────────────────────────────────────────────────

def run_factor_model():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    excess_returns, factors = load_master()
    results_df, residuals_df, models = run_all_regressions(excess_returns, factors)
    run_regression_sanity_checks(results_df)
    print_full_summary(models, ticker="AAPL")
    rolling_df = compute_rolling_betas(excess_returns, factors)

    return results_df, residuals_df, models, rolling_df


if __name__ == "__main__":
    results_df, residuals_df, models, rolling_df = run_factor_model()
    print("Phase 2 complete. Factor loadings ready for risk attribution.")