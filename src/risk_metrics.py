import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data():
    """
    Loads everything produced by Phases 1 and 2:
      - master_dataset:  aligned returns + factors
      - factor_loadings: OLS betas for each stock
      - residuals:       idiosyncratic return component per stock
    """
    master = pd.read_csv(
        f"{PROCESSED_DIR}/master_dataset.csv",
        index_col="Date", parse_dates=True
    )
    loadings = pd.read_csv(
        f"{PROCESSED_DIR}/factor_loadings.csv",
        index_col="Ticker"
    )
    residuals = pd.read_csv(
        f"{PROCESSED_DIR}/residuals.csv",
        index_col="Date", parse_dates=True
    )

    factor_cols = ["MKT_RF", "SMB", "HML", "MOM"]
    stock_cols  = [c for c in master.columns if c.endswith("_excess")]

    excess_returns = master[stock_cols].copy()
    excess_returns.columns = [c.replace("_excess", "") for c in stock_cols]
    factors = master[factor_cols].copy()

    print(f"Loaded: {excess_returns.shape[0]} days, {excess_returns.shape[1]} stocks")
    return excess_returns, factors, loadings, residuals


def equal_weights(tickers):
    """Returns an equal-weight vector for the given tickers."""
    n = len(tickers)
    return pd.Series(1 / n, index=tickers)


# ── Part 1: Risk Decomposition ────────────────────────────────────────────────

def compute_risk_decomposition(excess_returns, factors, loadings, residuals, weights=None):
    """
    Decomposes portfolio variance into:
      - Systematic variance:   driven by exposure to FF factors
      - Idiosyncratic variance: company-specific, unexplained by factors

    Method:
      Total variance       = w' Σ w          (full covariance matrix)
      Systematic variance  = w' B Σ_F B' w   (factor covariance contribution)
      Idiosyncratic var    = w' D w           (residual variance, diagonal matrix)

    Where:
      B     = matrix of factor loadings (n_stocks × n_factors)
      Σ_F   = covariance matrix of factors
      D     = diagonal matrix of residual variances
      w     = portfolio weight vector
    """
    tickers = excess_returns.columns.tolist()

    if weights is None:
        weights = equal_weights(tickers)

    # ── Stock-level decomposition ─────────────────────────────────────────
    factor_cols  = ["Beta_MKT", "Beta_SMB", "Beta_HML", "Beta_MOM"]
    B            = loadings[factor_cols].values           # (13 × 4)
    factor_cov   = factors.cov().values                   # (4 × 4)
    resid_vars   = residuals.var().values                 # (13,) diagonal

    # Systematic covariance matrix: B @ Σ_F @ B'
    systematic_cov = B @ factor_cov @ B.T                # (13 × 13)

    # Total covariance matrix (sample)
    total_cov    = excess_returns.cov().values            # (13 × 13)

    # Idiosyncratic: diagonal only (each stock's residual variance)
    idio_cov     = np.diag(resid_vars)                   # (13 × 13)

    # ── Portfolio-level decomposition ─────────────────────────────────────
    w = weights.values

    port_total_var  = float(w @ total_cov @ w)
    port_sys_var    = float(w @ systematic_cov @ w)
    port_idio_var   = float(w @ idio_cov @ w)

    port_total_vol  = np.sqrt(port_total_var)
    port_sys_vol    = np.sqrt(port_sys_var)
    port_idio_vol   = np.sqrt(port_idio_var)

    sys_pct  = port_sys_var  / port_total_var * 100
    idio_pct = port_idio_var / port_total_var * 100

    # ── Stock-level systematic vs idiosyncratic share ─────────────────────
    stock_decomp = []
    for i, ticker in enumerate(tickers):
        stock_total_var = total_cov[i, i]
        stock_sys_var   = systematic_cov[i, i]
        stock_idio_var  = idio_cov[i, i]
        stock_decomp.append({
            "Ticker"       : ticker,
            "Total_Vol_Ann": round(np.sqrt(stock_total_var) * np.sqrt(252), 4),
            "Sys_Vol_Ann"  : round(np.sqrt(stock_sys_var)   * np.sqrt(252), 4),
            "Idio_Vol_Ann" : round(np.sqrt(stock_idio_var)  * np.sqrt(252), 4),
            "Sys_Pct"      : round(stock_sys_var  / stock_total_var * 100, 1),
            "Idio_Pct"     : round(stock_idio_var / stock_total_var * 100, 1),
        })

    stock_decomp_df = pd.DataFrame(stock_decomp).set_index("Ticker")

    print("\n── Risk Decomposition (Stock Level) ─────────────────────────────")
    print(stock_decomp_df.to_string())
    print(f"\n── Portfolio-Level Decomposition (Equal Weight) ─────────────────")
    print(f"  Total portfolio volatility (ann.):       {port_total_vol * np.sqrt(252):.4f}")
    print(f"  Systematic volatility (ann.):            {port_sys_vol   * np.sqrt(252):.4f}")
    print(f"  Idiosyncratic volatility (ann.):         {port_idio_vol  * np.sqrt(252):.4f}")
    print(f"  Systematic share of variance:            {sys_pct:.1f}%")
    print(f"  Idiosyncratic share of variance:         {idio_pct:.1f}%")

    results = {
        "stock_decomp"   : stock_decomp_df,
        "port_total_vol" : port_total_vol,
        "port_sys_vol"   : port_sys_vol,
        "port_idio_vol"  : port_idio_vol,
        "sys_pct"        : sys_pct,
        "idio_pct"       : idio_pct,
        "total_cov"      : total_cov,
        "systematic_cov" : systematic_cov,
    }

    stock_decomp_df.to_csv(f"{PROCESSED_DIR}/risk_decomposition.csv")
    return results


# ── Part 2: VaR and CVaR ──────────────────────────────────────────────────────

def compute_portfolio_returns(excess_returns, weights=None):
    """Computes daily portfolio returns given weights."""
    tickers = excess_returns.columns.tolist()
    if weights is None:
        weights = equal_weights(tickers)
    port_returns = excess_returns[tickers].dot(weights)
    return port_returns


def historical_var_cvar(port_returns, confidence=0.95):
    """
    Historical simulation VaR and CVaR.

    VaR:  The loss threshold not exceeded (1-confidence)% of days.
          e.g. at 95%: the worst 5% of days are beyond this point.
    CVaR: The average loss on the days beyond VaR (Expected Shortfall).
          More conservative and preferred by regulators (Basel III).

    Sign convention: VaR and CVaR are reported as positive loss numbers.
    """
    sorted_returns = port_returns.sort_values()
    cutoff_idx     = int(np.floor((1 - confidence) * len(sorted_returns)))
    var            = -sorted_returns.iloc[cutoff_idx]
    cvar           = -sorted_returns.iloc[:cutoff_idx].mean()
    return float(var), float(cvar)


def parametric_var_cvar(port_returns, confidence=0.95):
    """
    Parametric (variance-covariance) VaR and CVaR.
    Assumes returns are normally distributed.

    VaR  = -(μ - z * σ)    where z = norm.ppf(1 - confidence)
    CVaR = -(μ - σ * φ(z) / (1 - confidence))
           where φ is the standard normal PDF

    The gap between historical and parametric CVaR is your fat-tail premium —
    how much the normality assumption underestimates real tail risk.
    """
    mu    = port_returns.mean()
    sigma = port_returns.std()
    z     = stats.norm.ppf(1 - confidence)
    var   = -(mu + z * sigma)
    # CVaR for normal distribution has a closed form
    cvar  = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
    return float(var), float(cvar)


def compute_var_cvar_full(excess_returns, weights=None):
    """
    Computes VaR and CVaR at both 95% and 99% confidence,
    using both historical and parametric methods.
    Also annualizes using the square-root-of-time rule.
    """
    port_returns = compute_portfolio_returns(excess_returns, weights)

    confidences  = [0.95, 0.99]
    var_results  = []

    print("\n── VaR and CVaR Summary ─────────────────────────────────────────")
    print(f"  Portfolio mean daily return:  {port_returns.mean():.5f}")
    print(f"  Portfolio daily volatility:   {port_returns.std():.5f}")
    print(f"  Portfolio ann. volatility:    {port_returns.std() * np.sqrt(252):.4f}")
    print()
    print(f"  {'Method':<14} {'Conf':<6} {'Daily VaR':>10} {'Daily CVaR':>11} "
          f"{'Ann. VaR':>10} {'Ann. CVaR':>11}")
    print(f"  {'-'*65}")

    for conf in confidences:
        h_var,  h_cvar  = historical_var_cvar(port_returns, conf)
        p_var,  p_cvar  = parametric_var_cvar(port_returns, conf)

        # Annualize using square-root-of-time rule
        h_var_ann  = h_var  * np.sqrt(252)
        h_cvar_ann = h_cvar * np.sqrt(252)
        p_var_ann  = p_var  * np.sqrt(252)
        p_cvar_ann = p_cvar * np.sqrt(252)

        print(f"  {'Historical':<14} {conf:<6.0%} {h_var:>10.5f} {h_cvar:>11.5f} "
              f"{h_var_ann:>10.4f} {h_cvar_ann:>11.4f}")
        print(f"  {'Parametric':<14} {conf:<6.0%} {p_var:>10.5f} {p_cvar:>11.5f} "
              f"{p_var_ann:>10.4f} {p_cvar_ann:>11.4f}")
        print()

        var_results.append({
            "Confidence"   : conf,
            "Hist_VaR"     : round(h_var,  5),
            "Hist_CVaR"    : round(h_cvar, 5),
            "Param_VaR"    : round(p_var,  5),
            "Param_CVaR"   : round(p_cvar, 5),
            "Hist_VaR_Ann" : round(h_var_ann,  4),
            "Hist_CVaR_Ann": round(h_cvar_ann, 4),
            "Param_VaR_Ann": round(p_var_ann,  4),
            "Param_CVaR_Ann":round(p_cvar_ann, 4),
        })

    var_df = pd.DataFrame(var_results)
    var_df.to_csv(f"{PROCESSED_DIR}/var_cvar.csv", index=False)

    # Fat-tail premium: how much historical CVaR exceeds parametric CVaR
    h95_cvar = var_results[0]["Hist_CVaR"]
    p95_cvar = var_results[0]["Param_CVaR"]
    premium  = (h95_cvar - p95_cvar) / p95_cvar * 100
    print(f"  Fat-tail premium at 95% (hist CVaR vs parametric CVaR): {premium:.1f}%")
    print(f"  → Parametric model underestimates tail risk by this amount")

    return var_df, port_returns


# ── Part 3: Stress Testing ────────────────────────────────────────────────────

STRESS_PERIODS = {
    "2008 Financial Crisis" : ("2008-09-01", "2009-03-31"),
    "COVID Crash"           : ("2020-02-01", "2020-04-30"),
    "2022 Rate Hikes"       : ("2022-01-01", "2022-12-31"),
    "Full Period"           : (None, None),   # baseline for comparison
}


def run_stress_tests(excess_returns, weights=None):
    """
    Re-computes portfolio risk metrics for each stress period.
    Comparing normal vs. crisis metrics tells you:
      1. How much volatility spikes during stress
      2. Whether your VaR estimates from calm periods are dangerously low
      3. Which crisis regime was hardest on this particular portfolio
    """
    port_returns = compute_portfolio_returns(excess_returns, weights)
    stress_results = []

    print("\n── Stress Test Results ──────────────────────────────────────────")
    print(f"  {'Period':<25} {'N Days':>7} {'Ann Vol':>8} {'MaxDD':>8} "
          f"{'Hist VaR95':>11} {'Hist CVaR95':>12}")
    print(f"  {'-'*75}")

    for period_name, (start, end) in STRESS_PERIODS.items():
        if start is None:
            subset = port_returns
        else:
            subset = port_returns.loc[start:end]

        if len(subset) < 20:
            print(f"  {period_name:<25} insufficient data — skipping")
            continue

        ann_vol    = subset.std() * np.sqrt(252)
        var95, _   = historical_var_cvar(subset, 0.95)
        cvar95, _  = historical_var_cvar(subset, 0.95)
        _, cvar95  = historical_var_cvar(subset, 0.95)

        # Maximum drawdown
        cumulative = (1 + subset).cumprod()
        rolling_max = cumulative.cummax()
        drawdown    = (cumulative - rolling_max) / rolling_max
        max_dd      = drawdown.min()

        # Total period return
        total_return = (1 + subset).prod() - 1

        print(f"  {period_name:<25} {len(subset):>7} {ann_vol:>8.4f} {max_dd:>8.4f} "
              f"{var95:>11.5f} {cvar95:>12.5f}")

        stress_results.append({
            "Period"       : period_name,
            "N_Days"       : len(subset),
            "Ann_Vol"      : round(ann_vol,      4),
            "Max_Drawdown" : round(max_dd,       4),
            "Total_Return" : round(total_return, 4),
            "Hist_VaR95"   : round(var95,        5),
            "Hist_CVaR95"  : round(cvar95,       5),
        })

    stress_df = pd.DataFrame(stress_results)
    stress_df.to_csv(f"{PROCESSED_DIR}/stress_tests.csv", index=False)
    print(f"\n  → Saved stress_tests.csv")

    # Key insight: how much worse was VaR during stress vs full period?
    full   = stress_df[stress_df["Period"] == "Full Period"]["Hist_VaR95"].values[0]
    covid  = stress_df[stress_df["Period"] == "COVID Crash"]["Hist_VaR95"].values
    hikes  = stress_df[stress_df["Period"] == "2022 Rate Hikes"]["Hist_VaR95"].values

    if len(covid) > 0:
        print(f"\n  VaR95 during COVID vs full period:      "
              f"{covid[0]:.5f} vs {full:.5f} "
              f"({(covid[0]/full - 1)*100:+.1f}%)")
    if len(hikes) > 0:
        print(f"  VaR95 during 2022 hikes vs full period: "
              f"{hikes[0]:.5f} vs {full:.5f} "
              f"({(hikes[0]/full - 1)*100:+.1f}%)")

    return stress_df


# ── Part 4: Ledoit-Wolf vs Sample Covariance (Stretch) ───────────────────────

def compare_covariance_estimators(excess_returns):
    """
    Compares two covariance matrix estimators:

    Sample covariance: standard, but noisy when T (days) is not >> N (stocks).
    With 1841 days and 13 stocks our T/N ratio is ~142 — reasonably large,
    so shrinkage won't change things dramatically. But it's worth showing
    you know the limitation exists.

    Ledoit-Wolf: shrinks the sample covariance toward a structured target
    (typically the identity matrix scaled), reducing estimation error.
    Commonly used in production portfolio construction.
    """
    from sklearn.covariance import LedoitWolf

    X = excess_returns.values

    # Sample covariance
    sample_cov = np.cov(X.T)

    # Ledoit-Wolf shrinkage
    lw         = LedoitWolf()
    lw.fit(X)
    lw_cov     = lw.covariance_
    shrinkage  = lw.shrinkage_

    # Compare implied portfolio volatilities (equal weight)
    n = X.shape[1]
    w = np.ones(n) / n

    vol_sample = np.sqrt(w @ sample_cov @ w) * np.sqrt(252)
    vol_lw     = np.sqrt(w @ lw_cov     @ w) * np.sqrt(252)

    print("\n── Ledoit-Wolf vs Sample Covariance ─────────────────────────────")
    print(f"  Shrinkage coefficient (α):               {shrinkage:.4f}")
    print(f"  Portfolio vol — sample covariance:       {vol_sample:.4f}")
    print(f"  Portfolio vol — Ledoit-Wolf:              {vol_lw:.4f}")
    print(f"  Difference:                              {abs(vol_sample - vol_lw):.4f}")
    print(f"\n  Interpretation:")
    print(f"  A shrinkage of {shrinkage:.2f} means Ledoit-Wolf pulls the sample")
    print(f"  covariance {shrinkage*100:.1f}% toward the structured target.")
    print(f"  With T/N ≈ {len(excess_returns)//n}, the sample matrix is relatively")
    print(f"  stable, so the correction is small but non-zero.")

    return sample_cov, lw_cov, shrinkage


# ── Main runner ───────────────────────────────────────────────────────────────

def run_risk_metrics():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    excess_returns, factors, loadings, residuals = load_data()
    weights = equal_weights(excess_returns.columns.tolist())

    decomp_results         = compute_risk_decomposition(
                                excess_returns, factors, loadings, residuals, weights)
    var_df, port_returns   = compute_var_cvar_full(excess_returns, weights)
    stress_df              = run_stress_tests(excess_returns, weights)
    sample_cov, lw_cov, _  = compare_covariance_estimators(excess_returns)

    print("\nPhase 3 complete. Risk metrics ready for visualization.")
    return decomp_results, var_df, stress_df, port_returns


if __name__ == "__main__":
    decomp_results, var_df, stress_df, port_returns = run_risk_metrics()