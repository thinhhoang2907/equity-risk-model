import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import zipfile
import io
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

TICKERS = [
    "AAPL", "MSFT", "GOOGL",   # Tech / Growth
    "JPM",  "BAC",             # Financials / Value
    "JNJ",  "PG",              # Healthcare & Consumer Staples / Defensive
    "XOM",  "CVX",             # Energy
    "CAT",  "DE",              # Industrials
    "AMZN", "NVDA",            # Tech / High-momentum
]

START_DATE = "2019-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"


# ── Step 1: Download stock prices ─────────────────────────────────────────────

def download_stock_prices(tickers, start, end):
    """
    Downloads adjusted close prices for all tickers via yfinance.
    auto_adjust=True gives split- and dividend-adjusted prices,
    which is what we need for clean return calculations.
    """
    print(f"Downloading price data for {len(tickers)} tickers...")

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    prices = raw["Close"]

    # Drop tickers missing more than 5% of observations
    threshold = 0.05 * len(prices)
    prices = prices.dropna(thresh=len(prices) - threshold, axis=1)

    # Forward-fill minor gaps (e.g. calendar mismatches)
    prices = prices.ffill()

    print(f"  → {prices.shape[0]} trading days, {prices.shape[1]} tickers")
    prices.to_csv(f"{RAW_DIR}/stock_prices.csv")
    return prices


# ── Step 2: Compute daily log returns ─────────────────────────────────────────

def compute_log_returns(prices):
    """
    Log returns: r_t = ln(P_t / P_{t-1})
    Preferred over simple returns because they're time-additive
    and more consistent with the academic FF literature.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    print(f"  → Log returns: {log_returns.shape}")
    log_returns.to_csv(f"{PROCESSED_DIR}/log_returns.csv")
    return log_returns


# ── Step 3: Download Fama-French factors ──────────────────────────────────────

def download_ff_factors(start, end):
    """
    Downloads FF 3-factor + momentum data using requests.
    Primary source: GitHub-hosted mirror of Ken French's data library.
    This sidesteps the DNS/firewall issues with Dartmouth's servers.
    """
    print("Downloading Fama-French factors...")

    def fetch_and_parse(url, col_names):
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        raw_text = z.read(csv_name).decode("utf-8", errors="ignore")

        # French CSVs have a text header — skip until we hit 8-digit dates
        lines = raw_text.splitlines()
        data_lines = []
        in_data = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if in_data:
                    break
                continue
            if stripped[:8].isdigit():
                in_data = True
                data_lines.append(stripped)
            elif in_data:
                break

        df = pd.read_csv(io.StringIO("\n".join(data_lines)), header=None)
        df.columns = col_names
        df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d")
        df = df.set_index("Date")
        return df

    # Try the direct Dartmouth URLs first, then a GitHub mirror
    ff3_urls = [
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip",
        "https://github.com/datasets/finance-vix/raw/main/data/ff3_daily.zip",
    ]
    mom_urls = [
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_daily_CSV.zip",
    ]

    ff3_df = None
    for url in ff3_urls:
        try:
            print(f"  Trying: {url[:60]}...")
            ff3_df = fetch_and_parse(
                url,
                ["Date", "MKT_RF", "SMB", "HML", "RF"]
            )
            print("  ✓ FF3 factors downloaded")
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    mom_df = None
    for url in mom_urls:
        try:
            print(f"  Trying: {url[:60]}...")
            mom_df = fetch_and_parse(url, ["Date", "MOM"])
            print("  ✓ Momentum factor downloaded")
            break
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    if ff3_df is None or mom_df is None:
        raise ConnectionError(
            "Could not download Fama-French data from any source.\n"
            "Check your internet connection or try on a different network."
        )

    factors = ff3_df.join(mom_df, how="inner")
    factors = factors.loc[start:end]
    factors = factors / 100
    factors = factors.apply(pd.to_numeric, errors="coerce").dropna()

    print(f"  → {factors.shape[0]} trading days of factor data")
    factors.to_csv(f"{RAW_DIR}/ff_factors.csv")
    return factors

# ── Step 4: Align and merge ───────────────────────────────────────────────────

def build_master_dataset(log_returns, factors):
    """
    Inner-joins stock returns with FF factors on trading date.
    Inner join ensures only days where both sources have data are kept,
    preventing misaligned rows from corrupting Phase 2 regressions.
    Also computes excess returns (stock return minus RF) which is
    the left-hand side variable in the Fama-French regression.
    """
    log_returns.index = pd.to_datetime(log_returns.index)
    factors.index     = pd.to_datetime(factors.index)

    merged = log_returns.join(factors, how="inner")

    print(f"  → Merged dataset: {merged.shape[0]} aligned trading days")
    print(f"     Date range: {merged.index[0].date()} → {merged.index[-1].date()}")

    rf         = merged["RF"]
    stock_cols = [c for c in merged.columns if c not in ["MKT_RF", "SMB", "HML", "RF", "MOM"]]

    excess_returns = merged[stock_cols].subtract(rf, axis=0)
    excess_returns.columns = [f"{c}_excess" for c in excess_returns.columns]

    master = pd.concat(
        [excess_returns, merged[["MKT_RF", "SMB", "HML", "MOM", "RF"]]],
        axis=1
    )
    master.to_csv(f"{PROCESSED_DIR}/master_dataset.csv")
    print(f"  → Saved to data/processed/master_dataset.csv")
    return master


# ── Step 5: Sanity checks ─────────────────────────────────────────────────────

def run_sanity_checks(master):
    """
    Run before moving to Phase 2. If any check looks wrong,
    debug the pipeline now rather than discovering it mid-regression.
    """
    print("\n── Sanity Checks ──────────────────────────────────────────────")

    stock_cols = [c for c in master.columns if c.endswith("_excess")]

    # 1. Missing values
    nulls = master[stock_cols].isnull().sum()
    if nulls.sum() == 0:
        print("✓ No missing values in excess returns")
    else:
        print(f"✗ Missing values found:\n{nulls[nulls > 0]}")

    # 2. Annualized stats — should look reasonable for US equities
    ann_return = master[stock_cols].mean() * 252
    ann_vol    = master[stock_cols].std() * np.sqrt(252)
    summary    = pd.DataFrame({"Ann. Return": ann_return, "Ann. Vol": ann_vol})
    print("\nAnnualized return and volatility:")
    print(summary.round(4).to_string())

    # 3. Factor descriptive stats
    factor_cols = ["MKT_RF", "SMB", "HML", "MOM"]
    print("\nFactor summary stats (daily, decimal):")
    print(master[factor_cols].describe().round(5).to_string())

    # 4. Average pairwise correlation
    corr     = master[stock_cols].corr()
    avg_corr = corr.values[np.triu_indices_from(corr.values, k=1)].mean()
    print(f"\nAverage pairwise stock correlation: {avg_corr:.3f}")
    print("  (Expected: 0.3–0.7 for a diversified US equity portfolio)")

    print("───────────────────────────────────────────────────────────────\n")


# ── Main runner ───────────────────────────────────────────────────────────────

def run_pipeline():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    prices      = download_stock_prices(TICKERS, START_DATE, END_DATE)
    log_returns = compute_log_returns(prices)
    factors     = download_ff_factors(START_DATE, END_DATE)
    master      = build_master_dataset(log_returns, factors)
    run_sanity_checks(master)

    return master


if __name__ == "__main__":
    master = run_pipeline()
    print("Phase 1 complete. Master dataset ready for modeling.")