import pandas as pd
import numpy as np
from pathlib import Path
from io import StringIO
import statsmodels.api as sm
from scipy.optimize import minimize
import matplotlib.pyplot as plt

DATA_DIR = Path("./Data")
OUTPUT_DIR = Path("./Output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SP100_FILE = DATA_DIR / "ESG_Ratings_SP100.csv"
DAX_FILE = DATA_DIR / "ESG_Ratings_DAX.csv"
PRICES_FILE = DATA_DIR / "Prices.csv"
FF_US_FILE = DATA_DIR / "FamaFrench_US_Monthly.csv"
FF_EU_FILE = DATA_DIR / "FamaFrench_Europe_Monthly.csv"

START_YEAR = 2015
END_YEAR = 2024
TOP_N = 10
BOTTOM_N = 10
RISK_PARITY_LOOKBACK = 252
TRADING_DAYS_PER_YEAR = 252

VERBOSE = True
EXPORT_RESULTS = True


def log(message):
    if VERBOSE:
        print(message)


def export_csv(df, filename):
    if EXPORT_RESULTS:
        df.to_csv(OUTPUT_DIR / filename, index=False, sep=";")


def load_esg_file(file_path, index_name):
    df = pd.read_csv(file_path, sep=";")
    df["Index"] = index_name
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")

    for col in ["Market Cap", "ESG Score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_prices(file_path):
    df = pd.read_csv(file_path, sep=";")
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y", errors="coerce")

    price_cols = [c for c in df.columns if c != "Date"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)
    return df


def clean_esg_data(df):
    df = df.copy()

    df = df.rename(columns={
        "Constituent Name": "Name",
        "Market Cap": "MarketCap",
        "ESG Score": "ESGScore",
        "ESG Grade": "ESGGrade",
        "Economic Sector": "EconomicSector"
    })

    df = df[df["Date"].dt.year.between(START_YEAR, END_YEAR)].copy()
    df = df.drop_duplicates(subset=["Index", "Date", "ISIN"]).reset_index(drop=True)

    return df


def clean_price_data(df):
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Date"]).reset_index(drop=True)

    return df, df.copy()


def get_rebalancing_dates(prices_df, start_year=START_YEAR, end_year=END_YEAR):
    rows = []

    trading_dates = pd.to_datetime(prices_df["Date"]).sort_values().unique()
    trading_dates = pd.DatetimeIndex(trading_dates)

    for year in range(start_year, end_year + 1):
        selection_date = pd.Timestamp(f"{year}-01-01")

        year_trading = trading_dates[trading_dates.year == year]
        if len(year_trading) == 0:
            continue

        rebalance_date = year_trading.min()
        holding_period_end = year_trading.max()

        rows.append({
            "Year": year,
            "SelectionDate": selection_date,
            "RebalanceDate": rebalance_date,
            "HoldingPeriodEnd": holding_period_end
        })

    reb = pd.DataFrame(rows).sort_values("Year").reset_index(drop=True)
    return reb


def build_investable_universe(esg_df, prices_df, rebalance_df, min_coverage=0.90):
    df = esg_df.copy()
    price_columns = set(prices_df.columns) - {"Date"}
    valid_selection_dates = set(rebalance_df["SelectionDate"])

    df = df[df["Date"].isin(valid_selection_dates)].copy()

    df = df[
        df["ESGScore"].notna() &
        df["MarketCap"].notna() &
        df["ISIN"].notna() &
        df["ISIN"].isin(price_columns)
    ].copy()

    df = df.merge(
        rebalance_df[["SelectionDate", "RebalanceDate", "HoldingPeriodEnd"]],
        left_on="Date",
        right_on="SelectionDate",
        how="left",
        validate="many_to_one"
    )

    eligible_rows = []

    for _, row in df.iterrows():
        isin = row["ISIN"]
        holding_start = row["RebalanceDate"]
        holding_end = row["HoldingPeriodEnd"]

        sub_prices = prices_df[
            (prices_df["Date"] >= holding_start) &
            (prices_df["Date"] <= holding_end)
        ][["Date", isin]].copy()

        if sub_prices.empty:
            continue

        total_obs = len(sub_prices)
        valid_obs = sub_prices[isin].notna().sum()

        coverage = valid_obs / total_obs if total_obs > 0 else 0.0

        if coverage >= min_coverage:
            eligible_rows.append(row)

    if not eligible_rows:
        return pd.DataFrame(columns=df.columns.tolist() + ["Eligible"])

    result = pd.DataFrame(eligible_rows).copy()
    result["Eligible"] = True

    result = result.drop(columns=["SelectionDate"], errors="ignore")
    return result.reset_index(drop=True)


def select_top_bottom_portfolios(universe_df, top_n=TOP_N, bottom_n=BOTTOM_N):
    selections = []

    for (index_name, date), grp in universe_df.groupby(["Index", "Date"]):
        grp = grp.copy()

        top_grp = (
            grp.sort_values(
                by=["ESGScore", "MarketCap", "ISIN"],
                ascending=[False, False, True]
            )
            .head(top_n)
            .copy()
        )
        top_grp["PortfolioSide"] = "Top"
        top_grp["ESGRankWithinSide"] = range(1, len(top_grp) + 1)

        bottom_grp = (
            grp.sort_values(
                by=["ESGScore", "MarketCap", "ISIN"],
                ascending=[True, False, True]
            )
            .head(bottom_n)
            .copy()
        )
        bottom_grp["PortfolioSide"] = "Bottom"
        bottom_grp["ESGRankWithinSide"] = range(1, len(bottom_grp) + 1)

        selections.append(top_grp)
        selections.append(bottom_grp)

    return pd.concat(selections, ignore_index=True).reset_index(drop=True)


def sanity_checks(selected_df):
    counts = (
        selected_df.groupby(["Index", selected_df["Date"].dt.year, "PortfolioSide"])["ISIN"]
        .count()
        .rename("Count")
        .reset_index()
    )

    bad = counts[~counts["Count"].eq(10)]

    overlaps = (
        selected_df.groupby(["Index", "Date", "ISIN"])["PortfolioSide"]
        .nunique()
        .reset_index(name="n_sides")
    )
    overlaps = overlaps[overlaps["n_sides"] > 1]

    if not bad.empty:
        print("\nWARNING: Some groups do not contain exactly 10 securities.")
        print(bad)

    if not overlaps.empty:
        print("\nWARNING: Some securities appear in both Top and Bottom at same date.")
        print(overlaps)

    if bad.empty and overlaps.empty:
        log("Sanity checks passed.")


def build_index_benchmark_returns(prices_df):
    benchmark_map = {
        "SP100": "S&P100",
        "DAX": "DAX"
    }

    results = []

    for index_name, ric in benchmark_map.items():
        if ric not in prices_df.columns:
            raise ValueError(f"Benchmark price column {ric} not found in Prices.csv")

        sub = prices_df[["Date", ric]].copy()
        sub = sub.rename(columns={ric: "Price"})
        sub["PortfolioReturn"] = sub["Price"].pct_change()
        sub = sub[
            (sub["Date"] >= pd.Timestamp("2015-01-01")) &
            (sub["Date"] <= pd.Timestamp("2024-12-31"))
        ].copy()
        sub = sub.dropna(subset=["PortfolioReturn"]).copy()

        sub["Index"] = index_name
        sub["PortfolioSide"] = "Benchmark"
        sub["Strategy"] = "Benchmark_MCap"

        results.append(
            sub[["Date", "Index", "PortfolioSide", "Strategy", "PortfolioReturn"]]
        )

    return pd.concat(results, ignore_index=True).sort_values(
        ["Index", "Date"]
    ).reset_index(drop=True)


def build_equal_benchmark_weights(universe_df):
    df = universe_df.copy()

    df["PortfolioSide"] = "Benchmark"
    df["Weight_Equal"] = (
        1 / df.groupby(["Index", "Date"])["ISIN"].transform("count")
    )

    return df.reset_index(drop=True)


def build_long_price_table(prices_df):
    benchmark_cols = {"S&P100", "DAX"}
    asset_cols = [c for c in prices_df.columns if c not in {"Date"} | benchmark_cols]

    long_df = prices_df.melt(
        id_vars="Date",
        value_vars=asset_cols,
        var_name="ISIN",
        value_name="Price"
    )

    long_df = long_df.dropna(subset=["Price"]).copy()
    long_df = long_df.sort_values(["ISIN", "Date"]).reset_index(drop=True)
    return long_df


def compute_daily_returns(long_price_df):
    df = long_price_df.copy()
    df["Return"] = df.groupby("ISIN")["Price"].pct_change()
    return df


def build_rebalanced_weights(selected_df):
    df = selected_df.copy()

    df["Weight_MCap"] = (
        df["MarketCap"] /
        df.groupby(["Index", "Date", "PortfolioSide"])["MarketCap"].transform("sum")
    )

    df["Weight_Equal"] = (
        1 / df.groupby(["Index", "Date", "PortfolioSide"])["ISIN"].transform("count")
    )

    return df


def attach_holding_periods(weights_df, rebalance_df):
    df = weights_df.copy()

    calendar = rebalance_df.copy().rename(columns={"SelectionDate": "Date"})

    df = df.merge(
        calendar[["Year", "Date", "RebalanceDate", "HoldingPeriodEnd"]],
        on="Date",
        how="left",
        validate="many_to_one"
    )

    return df


def get_lookback_returns_for_group(returns_df, rics, rebalance_date, lookback_days=RISK_PARITY_LOOKBACK):
    sub = returns_df[
        (returns_df["ISIN"].isin(rics)) &
        (returns_df["Date"] < rebalance_date)
    ].copy()

    if sub.empty:
        return pd.DataFrame()

    wide = sub.pivot(index="Date", columns="ISIN", values="Return").sort_index()

    available_rics = [ric for ric in rics if ric in wide.columns]
    wide = wide[available_rics].copy()

    if wide.empty:
        return pd.DataFrame()

    # Etwas Puffer, damit nach Cleaning noch genug Beobachtungen übrig bleiben
    wide = wide.tail(lookback_days + 40).copy()

    # Titel behalten, wenn sie ausreichend viele Beobachtungen haben
    min_non_missing = lookback_days
    valid_cols = wide.notna().sum()
    keep_cols = valid_cols[valid_cols >= min_non_missing].index.tolist()

    wide = wide[keep_cols].copy()

    if wide.empty:
        return pd.DataFrame()

    # Jetzt nur noch vollständige Tage für die verbleibenden Titel
    wide = wide.dropna(axis=0, how="any")

    if len(wide) < lookback_days:
        return pd.DataFrame()

    wide = wide.tail(lookback_days).copy()

    return wide


def portfolio_risk_contributions(weights, cov_matrix):
    portfolio_var = weights @ cov_matrix @ weights
    portfolio_vol = np.sqrt(portfolio_var)

    if portfolio_vol <= 0:
        return np.full_like(weights, np.nan)

    marginal_contrib = cov_matrix @ weights / portfolio_vol
    risk_contrib = weights * marginal_contrib

    return risk_contrib


def risk_parity_objective(weights, cov_matrix):
    rc = portfolio_risk_contributions(weights, cov_matrix)

    if np.any(np.isnan(rc)):
        return 1e10

    target_rc = np.sum(rc) / len(rc)
    return np.sum((rc - target_rc) ** 2)


def solve_risk_parity_weights(cov_matrix):
    n = cov_matrix.shape[0]

    # kleine Regularisierung für numerische Stabilität
    cov_matrix = cov_matrix + np.eye(n) * 1e-8

    def objective(weights):
        portfolio_var = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)

        if portfolio_vol <= 0:
            return 1e10

        marginal_contrib = cov_matrix @ weights / portfolio_vol
        risk_contrib = weights * marginal_contrib
        target = portfolio_vol / n

        return np.sum((risk_contrib - target) ** 2)

    # bessere Initialisierung: inverse-vol start
    vols = np.sqrt(np.diag(cov_matrix))
    inv_vol = 1 / vols
    init_w = inv_vol / inv_vol.sum()

    bounds = [(1e-8, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        objective,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False}
    )

    if not result.success:
        # fallback: inverse-volatility weights statt equal weights
        return init_w

    weights = np.clip(result.x, 0, None)

    if weights.sum() <= 0:
        return init_w

    return weights / weights.sum()


def build_risk_parity_weights(selected_df, returns_df, lookback_days=RISK_PARITY_LOOKBACK):
    all_weights = []

    grouped = selected_df.groupby(["Index", "RebalanceDate", "PortfolioSide"])

    for (index_name, rebalance_date, side), grp in grouped:
        grp = grp.copy()
        rics = grp["ISIN"].tolist()

        lookback_returns = get_lookback_returns_for_group(
            returns_df=returns_df,
            rics=rics,
            rebalance_date=rebalance_date,
            lookback_days=lookback_days
        )

        if lookback_returns.empty:
            continue

        usable_rics = lookback_returns.columns.tolist()

        if len(usable_rics) < 2:
            continue

        cov_matrix = lookback_returns.cov().values

        if cov_matrix.size == 0:
            continue

        rp_weights = solve_risk_parity_weights(cov_matrix)

        out = grp[grp["ISIN"].isin(usable_rics)].copy()
        out["Weight_RP"] = out["ISIN"].map(dict(zip(usable_rics, rp_weights)))
        out["RP_NumAssets"] = len(usable_rics)

        all_weights.append(out)

    if not all_weights:
        return pd.DataFrame(columns=list(selected_df.columns) + ["Weight_RP", "RP_NumAssets"])

    result = pd.concat(all_weights, ignore_index=True)
    return result.reset_index(drop=True)


def solve_long_only_mean_variance_weights(expected_returns, cov_matrix, risk_aversion=2.5):
    n = len(expected_returns)

    cov_matrix = cov_matrix + np.eye(n) * 1e-8

    def objective(weights):
        port_return = weights @ expected_returns
        port_var = weights @ cov_matrix @ weights
        return -(port_return - 0.5 * risk_aversion * port_var)

    init_w = np.repeat(1 / n, n)
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = minimize(
        objective,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False}
    )

    if not result.success:
        return init_w

    weights = np.clip(result.x, 0, None)

    if weights.sum() <= 0:
        return init_w

    return weights / weights.sum()


def build_black_litterman_weights(selected_df,
                                  returns_df,
                                  lookback_days=RISK_PARITY_LOOKBACK,
                                  tau=0.05,
                                  risk_aversion=2.5,
                                  view_strength_daily=0.0002):
    all_weights = []

    grouped = selected_df.groupby(["Index", "RebalanceDate", "PortfolioSide"])

    for (index_name, rebalance_date, side), grp in grouped:
        grp = grp.copy()
        rics = grp["ISIN"].tolist()

        lookback_returns = get_lookback_returns_for_group(
            returns_df=returns_df,
            rics=rics,
            rebalance_date=rebalance_date,
            lookback_days=lookback_days
        )

        if lookback_returns.empty:
            continue

        usable_rics = lookback_returns.columns.tolist()

        if len(usable_rics) < 2:
            continue

        sub_grp = grp[grp["ISIN"].isin(usable_rics)].copy()

        cov_matrix = lookback_returns.cov().values
        cov_matrix = cov_matrix + np.eye(len(usable_rics)) * 1e-8

        # Prior weights = market-cap weights innerhalb der verwendeten Titel
        market_caps = sub_grp.set_index("ISIN").loc[usable_rics, "MarketCap"].values.astype(float)
        if np.any(np.isnan(market_caps)) or market_caps.sum() <= 0:
            continue

        w_mkt = market_caps / market_caps.sum()

        # Implied equilibrium returns: pi = delta * Sigma * w_mkt
        pi = risk_aversion * (cov_matrix @ w_mkt)

        # ESG-based absolute views
        esg_scores = sub_grp.set_index("ISIN").loc[usable_rics, "ESGScore"].values.astype(float)

        if np.std(esg_scores) == 0:
            z_scores = np.zeros_like(esg_scores)
        else:
            z_scores = (esg_scores - np.mean(esg_scores)) / np.std(esg_scores)

        q = view_strength_daily * z_scores

        # Absolute views on each asset
        P = np.eye(len(usable_rics))

        # Standard BL choice: Omega = diag(tau * Sigma)
        omega = np.diag(np.diag(tau * cov_matrix))

        middle = np.linalg.inv(np.linalg.inv(tau * cov_matrix) + P.T @ np.linalg.inv(omega) @ P)
        posterior_mean = middle @ (
            np.linalg.inv(tau * cov_matrix) @ pi +
            P.T @ np.linalg.inv(omega) @ q
        )

        bl_weights = solve_long_only_mean_variance_weights(
            expected_returns=posterior_mean,
            cov_matrix=cov_matrix,
            risk_aversion=risk_aversion
        )

        out = sub_grp.copy()
        out["Weight_BL"] = out["ISIN"].map(dict(zip(usable_rics, bl_weights)))
        out["BL_NumAssets"] = len(usable_rics)

        all_weights.append(out)

    if not all_weights:
        return pd.DataFrame(columns=list(selected_df.columns) + ["Weight_BL", "BL_NumAssets"])

    result = pd.concat(all_weights, ignore_index=True)
    return result.reset_index(drop=True)


def build_annual_rebalanced_portfolio_returns(weights_df, returns_df, weight_column, strategy_name):
    all_results = []

    grouped = weights_df.groupby(["Index", "RebalanceDate", "PortfolioSide"])

    for (index_name, rebalance_date, side), grp in grouped:
        grp = grp.copy()

        holding_start = grp["RebalanceDate"].iloc[0]
        holding_end = grp["HoldingPeriodEnd"].iloc[0]

        rics = grp["ISIN"].tolist()
        start_weights = grp.set_index("ISIN")[weight_column].to_dict()

        sub = returns_df[
            (returns_df["ISIN"].isin(rics)) &
            (returns_df["Date"] >= holding_start) &
            (returns_df["Date"] <= holding_end)
        ].copy()

        if sub.empty:
            continue

        wide_returns = sub.pivot(index="Date", columns="ISIN", values="Return").sort_index()
        rics_in_panel = [ric for ric in rics if ric in wide_returns.columns]
        wide_returns = wide_returns[rics_in_panel].copy()
        
        min_required_obs = int(np.ceil(0.90 * len(wide_returns)))
        valid_cols = wide_returns.columns[wide_returns.notna().sum(axis=0) >= min_required_obs].tolist()
        wide_returns = wide_returns[valid_cols].copy()
        
        if wide_returns.empty:
            continue
        
        wide_returns = wide_returns.fillna(0.0)
        
        start_weights = {ric: w for ric, w in start_weights.items() if ric in valid_cols}
        
        weight_sum = sum(start_weights.values())
        if weight_sum <= 0:
            continue
        
        start_weights = {ric: w / weight_sum for ric, w in start_weights.items()}
        current_weights = np.array([start_weights[ric] for ric in valid_cols], dtype=float)
        portfolio_path = []

        for dt, row in wide_returns.iterrows():
            asset_returns = row.values.astype(float)
            port_ret = np.sum(current_weights * asset_returns)

            portfolio_path.append({
                "Date": dt,
                "Index": index_name,
                "PortfolioSide": side,
                "Strategy": strategy_name,
                "RebalanceDate": rebalance_date,
                "HoldingPeriodEnd": holding_end,
                "PortfolioReturn": port_ret
            })

            post_values = current_weights * (1 + asset_returns)
            total_value = post_values.sum()

            if total_value > 0:
                current_weights = post_values / total_value

        all_results.append(pd.DataFrame(portfolio_path))

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True).sort_values(
        ["Index", "PortfolioSide", "Strategy", "Date"]
    ).reset_index(drop=True)


def build_buy_and_hold_start_weights(selected_df, holding_start, holding_end, start_year=START_YEAR):
    start_date = pd.Timestamp(year=start_year, month=1, day=1)
    df = selected_df[selected_df["Date"] == start_date].copy()

    df["Weight_MCap"] = (
        df["MarketCap"] /
        df.groupby(["Index", "PortfolioSide"])["MarketCap"].transform("sum")
    )

    df["Weight_Equal"] = (
        1 / df.groupby(["Index", "PortfolioSide"])["ISIN"].transform("count")
    )

    df["RebalanceDate"] = holding_start
    df["HoldingPeriodEnd"] = holding_end

    return df.reset_index(drop=True)


def build_buy_and_hold_portfolio_returns(weights_df, returns_df, weight_column, strategy_name):
    all_results = []

    grouped = weights_df.groupby(["Index", "PortfolioSide"])

    for (index_name, side), grp in grouped:
        grp = grp.copy()

        holding_start = grp["RebalanceDate"].iloc[0]
        holding_end = grp["HoldingPeriodEnd"].iloc[0]

        rics = grp["ISIN"].tolist()
        start_weights = grp.set_index("ISIN")[weight_column].to_dict()

        sub = returns_df[
            (returns_df["ISIN"].isin(rics)) &
            (returns_df["Date"] >= holding_start) &
            (returns_df["Date"] <= holding_end)
        ].copy()

        if sub.empty:
            continue

        wide_returns = sub.pivot(index="Date", columns="ISIN", values="Return").sort_index()
        
        rics_in_panel = [ric for ric in rics if ric in wide_returns.columns]
        wide_returns = wide_returns[rics_in_panel].copy()
        
        min_required_obs = int(np.ceil(0.90 * len(wide_returns)))
        valid_cols = wide_returns.columns[wide_returns.notna().sum(axis=0) >= min_required_obs].tolist()
        wide_returns = wide_returns[valid_cols].copy()
        
        if wide_returns.empty:
            continue
        
        wide_returns = wide_returns.fillna(0.0)
        
        start_weights = {ric: w for ric, w in start_weights.items() if ric in valid_cols}
        
        weight_sum = sum(start_weights.values())
        if weight_sum <= 0:
            continue
        
        start_weights = {ric: w / weight_sum for ric, w in start_weights.items()}
        current_weights = np.array([start_weights[ric] for ric in valid_cols], dtype=float)
        portfolio_path = []

        for dt, row in wide_returns.iterrows():
            asset_returns = row.values.astype(float)
            port_ret = np.sum(current_weights * asset_returns)

            portfolio_path.append({
                "Date": dt,
                "Index": index_name,
                "PortfolioSide": side,
                "Strategy": strategy_name,
                "PortfolioReturn": port_ret
            })

            post_values = current_weights * (1 + asset_returns)
            total_value = post_values.sum()

            if total_value > 0:
                current_weights = post_values / total_value

        all_results.append(pd.DataFrame(portfolio_path))

    if not all_results:
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True).sort_values(
        ["Index", "PortfolioSide", "Strategy", "Date"]
    ).reset_index(drop=True)


def build_top_minus_bottom_returns(portfolio_returns_df):
    df = portfolio_returns_df.copy()

    df = df[df["PortfolioSide"].isin(["Top", "Bottom"])].copy()

    top = df[df["PortfolioSide"] == "Top"].copy()
    bottom = df[df["PortfolioSide"] == "Bottom"].copy()

    merge_cols = ["Date", "Index", "Strategy"]

    spread = top.merge(
        bottom,
        on=merge_cols,
        suffixes=("_Top", "_Bottom"),
        how="inner"
    )

    spread["PortfolioReturn"] = spread["PortfolioReturn_Top"] - spread["PortfolioReturn_Bottom"]
    spread["PortfolioSide"] = "TopMinusBottom"

    return spread[["Date", "Index", "PortfolioSide", "Strategy", "PortfolioReturn"]].copy()


def build_portfolio_minus_benchmark_returns(portfolio_returns_df):
    df = portfolio_returns_df.copy()
    results = []

    benchmark_map = {
        "Annual_Equal": "Benchmark_Equal",
        "BuyHold_Equal": "Benchmark_Equal",
        "Annual_MCap": "Benchmark_MCap",
        "BuyHold_MCap": "Benchmark_MCap",
        "Annual_RiskParity": "Benchmark_MCap",
        "Annual_BlackLitterman": "Benchmark_MCap"
    }

    for strategy, bench_strategy in benchmark_map.items():
        strat_df = df[df["Strategy"] == strategy].copy()
        bench_df = df[
            (df["Strategy"] == bench_strategy) &
            (df["PortfolioSide"] == "Benchmark")
        ].copy()

        merged = strat_df.merge(
            bench_df[["Date", "Index", "PortfolioReturn"]],
            on=["Date", "Index"],
            suffixes=("", "_Benchmark"),
            how="inner"
        )

        merged["PortfolioReturn"] = merged["PortfolioReturn"] - merged["PortfolioReturn_Benchmark"]
        merged["Strategy"] = strategy + "_MinusBenchmark"

        results.append(
            merged[["Date", "Index", "PortfolioSide", "Strategy", "PortfolioReturn"]]
        )

    return pd.concat(results, ignore_index=True).sort_values(
        ["Index", "PortfolioSide", "Strategy", "Date"]
    ).reset_index(drop=True)


def compute_annual_returns(portfolio_returns_df):
    df = portfolio_returns_df.copy()
    df["Year"] = df["Date"].dt.year

    annual = (
        df.groupby(["Index", "PortfolioSide", "Strategy", "Year"])["PortfolioReturn"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index(name="AnnualReturn")
    )

    return annual


def compute_monthly_portfolio_returns(daily_portfolio_df):
    df = daily_portfolio_df.copy()
    df["YearMonth"] = df["Date"].dt.to_period("M")

    monthly = (
        df.groupby(["Index", "PortfolioSide", "Strategy", "YearMonth"])["PortfolioReturn"]
        .apply(lambda x: (1 + x).prod() - 1)
        .reset_index(name="MonthlyReturn")
    )

    monthly["MonthEnd"] = monthly["YearMonth"].dt.to_timestamp(how="end").dt.normalize()
    return monthly


def load_ff_file(file_path, region):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        if "Mkt-RF" in line and "SMB" in line and "HML" in line and "RF" in line:
            header_idx = i
            break

    if header_idx is None:
        raise ValueError(f"No header row found in {file_path.name}")

    start_idx = None
    for i in range(header_idx + 1, len(lines)):
        first_token = lines[i].split(",")[0].strip()
        if len(first_token) == 6 and first_token.isdigit():
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(f"No monthly data start found in {file_path.name}")

    end_idx = None
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line == "":
            end_idx = i
            break

        first_token = line.split(",")[0].strip()
        if not (len(first_token) == 6 and first_token.isdigit()):
            end_idx = i
            break

    if end_idx is None:
        end_idx = len(lines)

    data_str = "".join(lines[start_idx:end_idx])

    df = pd.read_csv(
        StringIO(data_str),
        sep=",",
        header=None,
        names=["Date", "Mkt-RF", "SMB", "HML", "RF"]
    )

    for col in ["Date", "Mkt-RF", "SMB", "HML", "RF"]:
        df[col] = df[col].astype(str).str.strip()

    df = df.replace("-99.99", np.nan)
    df = df.replace("-999", np.nan)

    df["Date"] = pd.to_datetime(df["Date"] + "01", format="%Y%m%d", errors="coerce")
    df["YearMonth"] = df["Date"].dt.to_period("M")

    for col in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100.0

    df["FFRegion"] = region
    return df[["YearMonth", "Mkt-RF", "SMB", "HML", "RF", "FFRegion"]].dropna()


def attach_ff_factors(monthly_returns_df, ff_us_df, ff_eu_df):
    df = monthly_returns_df.copy()

    us_part = df[df["Index"] == "SP100"].merge(
        ff_us_df,
        on="YearMonth",
        how="left"
    )

    eu_part = df[df["Index"] == "DAX"].merge(
        ff_eu_df,
        on="YearMonth",
        how="left"
    )

    combined = pd.concat([us_part, eu_part], ignore_index=True)
    combined["ExcessReturn"] = combined["MonthlyReturn"] - combined["RF"]

    return combined


def run_ff3_regression(df):
    results = []

    grouped = df.groupby(["Index", "PortfolioSide", "Strategy"])

    for (index_name, side, strategy), grp in grouped:
        grp = grp.dropna(subset=["ExcessReturn", "Mkt-RF", "SMB", "HML"]).copy()

        if len(grp) < 12:
            continue

        y = grp["ExcessReturn"]
        X = grp[["Mkt-RF", "SMB", "HML"]]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

        results.append({
            "Index": index_name,
            "PortfolioSide": side,
            "Strategy": strategy,
            "NumMonths": len(grp),
            "Alpha_Monthly": model.params.get("const", np.nan),
            "Alpha_tstat": model.tvalues.get("const", np.nan),
            "Alpha_pvalue": model.pvalues.get("const", np.nan),
            "Beta_Mkt": model.params.get("Mkt-RF", np.nan),
            "Beta_SMB": model.params.get("SMB", np.nan),
            "Beta_HML": model.params.get("HML", np.nan),
            "R_squared": model.rsquared
        })

    return pd.DataFrame(results)


def filter_monthly_period(df, start_ym, end_ym):
    start = pd.Period(start_ym, freq="M")
    end = pd.Period(end_ym, freq="M")
    return df[(df["YearMonth"] >= start) & (df["YearMonth"] <= end)].copy()


def make_pretty_ff3_table(df):
    out = df.copy()

    alpha_monthly_raw = out["Alpha_Monthly"].copy()
    alpha_annual_raw = alpha_monthly_raw * 12

    out["Alpha_Monthly"] = (alpha_monthly_raw * 100).round(3)
    out["Alpha_Annualised"] = (alpha_annual_raw * 100).round(3)
    out["Alpha_tstat"] = out["Alpha_tstat"].round(3)
    out["Alpha_pvalue"] = out["Alpha_pvalue"].round(4)
    out["Beta_Mkt"] = out["Beta_Mkt"].round(3)
    out["Beta_SMB"] = out["Beta_SMB"].round(3)
    out["Beta_HML"] = out["Beta_HML"].round(3)
    out["R_squared"] = out["R_squared"].round(3)

    return out


def compute_max_drawdown(return_series):
    if return_series.empty:
        return np.nan

    wealth_index = (1 + return_series.fillna(0)).cumprod()
    running_max = wealth_index.cummax()
    drawdown = wealth_index / running_max - 1

    return drawdown.min()


def compute_historical_var(return_series, alpha=0.05):
    if return_series.empty:
        return np.nan

    q = return_series.quantile(alpha)
    return -q


def attach_daily_rf_to_portfolio_returns(daily_portfolio_df, ff_us_df, ff_eu_df):
    df = daily_portfolio_df.copy()
    df["YearMonth"] = df["Date"].dt.to_period("M")

    us = df[df["Index"] == "SP100"].merge(
        ff_us_df[["YearMonth", "RF"]],
        on="YearMonth",
        how="left"
    )

    eu = df[df["Index"] == "DAX"].merge(
        ff_eu_df[["YearMonth", "RF"]],
        on="YearMonth",
        how="left"
    )

    combined = pd.concat([us, eu], ignore_index=True)
    combined["RF_Daily"] = (1 + combined["RF"]) ** (1 / 21) - 1

    return combined


def compute_performance_metrics(df):
    results = []

    grouped = df.groupby(["Index", "PortfolioSide", "Strategy"])

    for (index_name, side, strategy), grp in grouped:
        grp = grp.sort_values("Date").copy()

        rets = grp["PortfolioReturn"].dropna()
        rf_daily = grp.loc[rets.index, "RF_Daily"].fillna(0.0)

        if rets.empty:
            continue

        n_obs = len(rets)
        cumulative_return = (1 + rets).prod() - 1
        annualised_return = (1 + cumulative_return) ** (TRADING_DAYS_PER_YEAR / n_obs) - 1

        daily_vol = rets.std(ddof=1)
        annualised_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR)

        excess_daily = rets - rf_daily
        excess_annualised_return = excess_daily.mean() * TRADING_DAYS_PER_YEAR

        sharpe_ratio = np.nan
        if annualised_vol > 0:
            sharpe_ratio = excess_annualised_return / annualised_vol

        var_95 = compute_historical_var(rets, alpha=0.05)
        max_drawdown = compute_max_drawdown(rets)

        results.append({
            "Index": index_name,
            "PortfolioSide": side,
            "Strategy": strategy,
            "NumObservations": n_obs,
            "CumulativeReturn": cumulative_return,
            "AnnualisedReturn": annualised_return,
            "AnnualisedVolatility": annualised_vol,
            "SharpeRatio": sharpe_ratio,
            "HistoricalVaR95": var_95,
            "MaxDrawdown": max_drawdown
        })

    return pd.DataFrame(results)


def filter_period(df, start_date, end_date):
    df = df.copy()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return df[(df["Date"] >= start) & (df["Date"] <= end)].copy()


def make_pretty_metrics_table(metrics_df):
    df = metrics_df.copy()

    pct_cols = [
        "CumulativeReturn",
        "AnnualisedReturn",
        "AnnualisedVolatility",
        "HistoricalVaR95",
        "MaxDrawdown"
    ]

    for col in pct_cols:
        df[col] = (df[col] * 100).round(2)

    df["SharpeRatio"] = df["SharpeRatio"].round(3)
    return df


sp100_raw = load_esg_file(SP100_FILE, "SP100")
dax_raw = load_esg_file(DAX_FILE, "DAX")
prices_raw = load_prices(PRICES_FILE)
esg_raw = pd.concat([sp100_raw, dax_raw], ignore_index=True)

esg = clean_esg_data(esg_raw)
prices_all, prices_returns = clean_price_data(prices_raw)
rebalancing_dates = get_rebalancing_dates(prices_all)
investable_universe = build_investable_universe(esg, prices_all, rebalancing_dates)
selected_portfolios = select_top_bottom_portfolios(investable_universe)
sanity_checks(selected_portfolios)

prices_long = build_long_price_table(prices_returns)
returns_long = compute_daily_returns(prices_long)

weights_rebalanced = build_rebalanced_weights(selected_portfolios)
weights_rp = build_risk_parity_weights(
    selected_df=selected_portfolios,
    returns_df=returns_long,
    lookback_days=RISK_PARITY_LOOKBACK
)

benchmark_equal_weights = build_equal_benchmark_weights(investable_universe)

weights_bl = build_black_litterman_weights(
    selected_df=selected_portfolios,
    returns_df=returns_long,
    lookback_days=RISK_PARITY_LOOKBACK,
    tau=0.05,
    risk_aversion=2.5,
    view_strength_daily=0.0002
)

annual_mcap_returns = build_annual_rebalanced_portfolio_returns(
    weights_df=weights_rebalanced,
    returns_df=returns_long,
    weight_column="Weight_MCap",
    strategy_name="Annual_MCap"
)

annual_equal_returns = build_annual_rebalanced_portfolio_returns(
    weights_df=weights_rebalanced,
    returns_df=returns_long,
    weight_column="Weight_Equal",
    strategy_name="Annual_Equal"
)

benchmark_mcap_returns = build_index_benchmark_returns(prices_returns)

benchmark_equal_returns = build_annual_rebalanced_portfolio_returns(
    weights_df=benchmark_equal_weights,
    returns_df=returns_long,
    weight_column="Weight_Equal",
    strategy_name="Benchmark_Equal"
)

buyhold_start = rebalancing_dates.loc[
    rebalancing_dates["Year"] == START_YEAR, "RebalanceDate"
].iloc[0]

buyhold_end = rebalancing_dates.loc[
    rebalancing_dates["Year"] == END_YEAR, "HoldingPeriodEnd"
].iloc[0]

weights_buyhold = build_buy_and_hold_start_weights(
    selected_portfolios,
    holding_start=buyhold_start,
    holding_end=buyhold_end
)

buyhold_mcap_returns = build_buy_and_hold_portfolio_returns(
    weights_df=weights_buyhold,
    returns_df=returns_long,
    weight_column="Weight_MCap",
    strategy_name="BuyHold_MCap"
)

annual_rp_returns = build_annual_rebalanced_portfolio_returns(
    weights_df=weights_rp,
    returns_df=returns_long,
    weight_column="Weight_RP",
    strategy_name="Annual_RiskParity"
)

annual_bl_returns = build_annual_rebalanced_portfolio_returns(
    weights_df=weights_bl,
    returns_df=returns_long,
    weight_column="Weight_BL",
    strategy_name="Annual_BlackLitterman"
)

buyhold_equal_returns = build_buy_and_hold_portfolio_returns(
    weights_df=weights_buyhold,
    returns_df=returns_long,
    weight_column="Weight_Equal",
    strategy_name="BuyHold_Equal"
)

all_portfolio_returns = pd.concat(
    [
        annual_mcap_returns,
        annual_equal_returns,
        annual_rp_returns,
        annual_bl_returns,
        buyhold_mcap_returns,
        buyhold_equal_returns,
        benchmark_mcap_returns,
        benchmark_equal_returns
    ],
    ignore_index=True,
    sort=False
)

all_portfolio_returns = all_portfolio_returns.sort_values(
    ["Index", "PortfolioSide", "Strategy", "Date"]
).reset_index(drop=True)

top_minus_bottom_returns = build_top_minus_bottom_returns(all_portfolio_returns)

portfolio_minus_benchmark_returns = build_portfolio_minus_benchmark_returns(all_portfolio_returns)

all_portfolio_returns_extended = pd.concat(
    [
        all_portfolio_returns,
        top_minus_bottom_returns,
        portfolio_minus_benchmark_returns
    ],
    ignore_index=True,
    sort=False
)

all_portfolio_returns_extended = all_portfolio_returns_extended.sort_values(
    ["Index", "PortfolioSide", "Strategy", "Date"]
).reset_index(drop=True)

annual_returns = compute_annual_returns(all_portfolio_returns_extended)

monthly_portfolio_returns = compute_monthly_portfolio_returns(all_portfolio_returns_extended)
ff_us = load_ff_file(FF_US_FILE, "US")
ff_eu = load_ff_file(FF_EU_FILE, "EU")
ff_regression_data = attach_ff_factors(monthly_portfolio_returns, ff_us, ff_eu)

ff_regression_data = ff_regression_data[
    ff_regression_data["PortfolioSide"] != "Benchmark"
].copy()

ff_data_full = ff_regression_data.copy()
ff_data_2015_2019 = filter_monthly_period(ff_regression_data, "2015-01", "2019-12")
ff_data_2020_2024 = filter_monthly_period(ff_regression_data, "2020-01", "2024-12")

ff3_full = run_ff3_regression(ff_data_full)
ff3_full["Period"] = "2015-2024"

ff3_2015_2019 = run_ff3_regression(ff_data_2015_2019)
ff3_2015_2019["Period"] = "2015-2019"

ff3_2020_2024 = run_ff3_regression(ff_data_2020_2024)
ff3_2020_2024["Period"] = "2020-2024"

ff3_all = pd.concat([ff3_full, ff3_2015_2019, ff3_2020_2024], ignore_index=True)

ff3_all = ff3_all[
    [
        "Period", "Index", "PortfolioSide", "Strategy", "NumMonths",
        "Alpha_Monthly", "Alpha_tstat", "Alpha_pvalue",
        "Beta_Mkt", "Beta_SMB", "Beta_HML", "R_squared"
    ]
].sort_values(["Period", "Index", "PortfolioSide", "Strategy"]).reset_index(drop=True)

ff3_pretty = make_pretty_ff3_table(ff3_all)

returns_full = attach_daily_rf_to_portfolio_returns(all_portfolio_returns_extended.copy(), ff_us, ff_eu)
returns_2015_2019 = attach_daily_rf_to_portfolio_returns(
    filter_period(all_portfolio_returns_extended, "2015-01-01", "2019-12-31"),
    ff_us,
    ff_eu
)
returns_2020_2024 = attach_daily_rf_to_portfolio_returns(
    filter_period(all_portfolio_returns_extended, "2020-01-01", "2024-12-31"),
    ff_us,
    ff_eu
)

metrics_full = compute_performance_metrics(returns_full)
metrics_full["Period"] = "2015-2024"

metrics_2015_2019 = compute_performance_metrics(returns_2015_2019)
metrics_2015_2019["Period"] = "2015-2019"

metrics_2020_2024 = compute_performance_metrics(returns_2020_2024)
metrics_2020_2024["Period"] = "2020-2024"

all_metrics = pd.concat(
    [metrics_full, metrics_2015_2019, metrics_2020_2024],
    ignore_index=True
)

all_metrics = all_metrics[
    [
        "Period", "Index", "PortfolioSide", "Strategy", "NumObservations",
        "CumulativeReturn", "AnnualisedReturn", "AnnualisedVolatility",
        "SharpeRatio", "HistoricalVaR95", "MaxDrawdown"
    ]
].sort_values(["Period", "Index", "PortfolioSide", "Strategy"]).reset_index(drop=True)

pretty_metrics = make_pretty_metrics_table(all_metrics)

export_csv(selected_portfolios, "selected_esg_portfolios.csv")
export_csv(all_portfolio_returns, "all_portfolio_returns.csv")
export_csv(top_minus_bottom_returns, "top_minus_bottom_returns.csv")
export_csv(portfolio_minus_benchmark_returns, "portfolio_minus_benchmark_returns.csv")
export_csv(all_portfolio_returns_extended, "all_portfolio_returns_extended.csv")
export_csv(annual_returns, "annual_returns.csv")
export_csv(pretty_metrics, "all_metrics_pretty.csv")
export_csv(ff3_pretty, "ff3_all_pretty.csv")
export_csv(weights_rp, "weights_risk_parity.csv")
export_csv(weights_bl, "weights_black_litterman.csv")

CHART_DIR = OUTPUT_DIR / "Charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def build_cumulative_returns(df):
    out = df.copy()
    out = out.sort_values(["Index", "PortfolioSide", "Strategy", "Date"]).copy()
    out["CumulativeWealth"] = (
        out.groupby(["Index", "PortfolioSide", "Strategy"])["PortfolioReturn"]
        .transform(lambda x: (1 + x).cumprod())
    )
    return out


def build_drawdown_series(df):
    out = df.copy().sort_values(["Index", "PortfolioSide", "Strategy", "Date"]).copy()

    def _drawdown(x):
        wealth = (1 + x).cumprod()
        running_max = wealth.cummax()
        return wealth / running_max - 1

    out["Drawdown"] = (
        out.groupby(["Index", "PortfolioSide", "Strategy"])["PortfolioReturn"]
        .transform(_drawdown)
    )
    return out


def plot_line_group(df, index_name, title, strategies_to_keep, value_col, filename):
    sub = df[
        (df["Index"] == index_name) &
        (df["Strategy"].isin(strategies_to_keep))
    ].copy()

    if sub.empty:
        return

    sub["Label"] = sub.apply(make_strategy_label, axis=1)

    plt.figure(figsize=(12, 7))

    for label, grp in sub.groupby("Label"):
        grp = grp.sort_values("Date")
        plt.plot(grp["Date"], grp[value_col], label=label, linewidth=0.8)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.legend(fontsize=9)
    save_plot(filename)


def make_strategy_label(row):
    side = row["PortfolioSide"]
    strategy = row["Strategy"]

    if strategy == "Benchmark_Equal":
        return "Benchmark"
    if strategy == "Benchmark_MCap":
        return "Benchmark"

    if strategy == "Annual_Equal" and side == "Top":
        return "Top | Annual"
    if strategy == "Annual_Equal" and side == "Bottom":
        return "Bottom | Annual"
    if strategy == "BuyHold_Equal" and side == "Top":
        return "Top | Buy & Hold"
    if strategy == "BuyHold_Equal" and side == "Bottom":
        return "Bottom | Buy & Hold"

    if strategy == "Annual_MCap" and side == "Top":
        return "Top | Annual"
    if strategy == "Annual_MCap" and side == "Bottom":
        return "Bottom | Annual"
    if strategy == "BuyHold_MCap" and side == "Top":
        return "Top | Buy & Hold"
    if strategy == "BuyHold_MCap" and side == "Bottom":
        return "Bottom | Buy & Hold"

    if strategy == "Annual_RiskParity" and side == "Top":
        return "Top"
    if strategy == "Annual_RiskParity" and side == "Bottom":
        return "Bottom"

    if strategy == "Annual_BlackLitterman" and side == "Top":
        return "Top"
    if strategy == "Annual_BlackLitterman" and side == "Bottom":
        return "Bottom"

    return f"{side} | {strategy}"


def build_summary_table(metrics_df, ff3_df, index_name, family):
    family_map = {
        "equal": ["Annual_Equal", "BuyHold_Equal", "Benchmark_Equal"],
        "mcap": ["Annual_MCap", "BuyHold_MCap", "Benchmark_MCap"],
        "rp": ["Annual_RiskParity", "Benchmark_MCap"],
        "bl": ["Annual_BlackLitterman", "Benchmark_MCap"]
    }

    label_order_map = {
        "equal": ["Top | Annual", "Bottom | Annual", "Top | Buy & Hold", "Bottom | Buy & Hold", "Benchmark"],
        "mcap": ["Top | Annual", "Bottom | Annual", "Top | Buy & Hold", "Bottom | Buy & Hold", "Benchmark"],
        "rp": ["Top", "Bottom", "Benchmark"],
        "bl": ["Top", "Bottom", "Benchmark"]
    }

    strategies = family_map[family]
    label_order = label_order_map[family]

    metric_sub = metrics_df[
        (metrics_df["Index"] == index_name) &
        (metrics_df["Period"] == "2015-2024") &
        (metrics_df["Strategy"].isin(strategies))
    ].copy()

    ff3_sub = ff3_df[
        (ff3_df["Index"] == index_name) &
        (ff3_df["Period"] == "2015-2024") &
        (ff3_df["Strategy"].isin(strategies))
    ].copy()

    metric_sub["Label"] = metric_sub.apply(make_strategy_label, axis=1)
    ff3_sub["Label"] = ff3_sub.apply(make_strategy_label, axis=1)
    
    metric_sub = metric_sub[metric_sub["PortfolioSide"] != "TopMinusBottom"].copy()
    ff3_sub = ff3_sub[ff3_sub["PortfolioSide"] != "TopMinusBottom"].copy()

    table = metric_sub.merge(
        ff3_sub[["Label", "Alpha_Annualised", "Alpha_tstat", "Alpha_pvalue"]],
        on="Label",
        how="left"
    )

    for col in ["Alpha_Annualised", "Alpha_tstat", "Alpha_pvalue"]:
        table[col] = table[col].astype(object)
        table.loc[table[col].isna(), col] = "n/a"

    table = table[
        [
            "Label",
            "AnnualisedReturn",
            "AnnualisedVolatility",
            "SharpeRatio",
            "HistoricalVaR95",
            "MaxDrawdown",
            "Alpha_Annualised",
            "Alpha_tstat",
            "Alpha_pvalue"
        ]
    ].copy()

    table["Label"] = pd.Categorical(table["Label"], categories=label_order, ordered=True)
    table = table.sort_values("Label").reset_index(drop=True)

    return table

summary_tables = {}

for index_name in sorted(pretty_metrics["Index"].unique()):
    for family in ["equal", "mcap", "rp", "bl"]:
        table = build_summary_table(pretty_metrics, ff3_pretty, index_name, family)
        summary_tables[f"{index_name.lower()}_{family}_summary.csv"] = table

for filename, table in summary_tables.items():
    export_csv(table, filename)
cumulative_returns = build_cumulative_returns(all_portfolio_returns)
drawdown_df = build_drawdown_series(all_portfolio_returns)

for index_name in sorted(cumulative_returns["Index"].unique()):
    plot_line_group(
        df=cumulative_returns,
        index_name=index_name,
        title=f"{index_name} - Equal Weighted",
        strategies_to_keep=["Annual_Equal", "BuyHold_Equal", "Benchmark_Equal"],
        value_col="CumulativeWealth",
        filename=f"{index_name.lower()}_equal_weighted_cumulative.png"
    )

    plot_line_group(
        df=cumulative_returns,
        index_name=index_name,
        title=f"{index_name} - Market-Cap Weighted",
        strategies_to_keep=["Annual_MCap", "BuyHold_MCap", "Benchmark_MCap"],
        value_col="CumulativeWealth",
        filename=f"{index_name.lower()}_market_cap_weighted_cumulative.png"
    )

    plot_line_group(
        df=cumulative_returns,
        index_name=index_name,
        title=f"{index_name} - Risk Parity",
        strategies_to_keep=["Annual_RiskParity", "Benchmark_MCap"],
        value_col="CumulativeWealth",
        filename=f"{index_name.lower()}_risk_parity_cumulative.png"
    )

    plot_line_group(
        df=cumulative_returns,
        index_name=index_name,
        title=f"{index_name} - Black-Litterman",
        strategies_to_keep=["Annual_BlackLitterman", "Benchmark_MCap"],
        value_col="CumulativeWealth",
        filename=f"{index_name.lower()}_black_litterman_cumulative.png"
    )

    plot_line_group(
        df=drawdown_df,
        index_name=index_name,
        title=f"{index_name} - Equal Weighted Drawdowns",
        strategies_to_keep=["Annual_Equal", "BuyHold_Equal", "Benchmark_Equal"],
        value_col="Drawdown",
        filename=f"{index_name.lower()}_equal_weighted_drawdown.png"
    )

    plot_line_group(
        df=drawdown_df,
        index_name=index_name,
        title=f"{index_name} - Market-Cap Weighted Drawdowns",
        strategies_to_keep=["Annual_MCap", "BuyHold_MCap", "Benchmark_MCap"],
        value_col="Drawdown",
        filename=f"{index_name.lower()}_market_cap_weighted_drawdown.png"
    )

    plot_line_group(
        df=drawdown_df,
        index_name=index_name,
        title=f"{index_name} - Risk Parity Drawdowns",
        strategies_to_keep=["Annual_RiskParity", "Benchmark_MCap"],
        value_col="Drawdown",
        filename=f"{index_name.lower()}_risk_parity_drawdown.png"
    )

    plot_line_group(
        df=drawdown_df,
        index_name=index_name,
        title=f"{index_name} - Black-Litterman Drawdowns",
        strategies_to_keep=["Annual_BlackLitterman", "Benchmark_MCap"],
        value_col="Drawdown",
        filename=f"{index_name.lower()}_black_litterman_drawdown.png"
    )

if VERBOSE:
    print("CSVs exported to Output")
    print("Charts exported to Output/Charts")
    print("Run completed successfully.")