"""Adverse-scenario stress forecast of US house prices.

Multi-step projection of HPI under a fixed adverse macro scenario, with
uncertainty from a normal ("Laplace") approximation to the posterior of the
model coefficients.

Why a growth model. HPI is very close to a random walk, so recursively forecasting
a model of the price *level* is numerically unstable (it blows up). The stable,
standard approach is to model the quarterly *growth rate* (log return) as a
function of macro conditions and then cumulate the projected growth. That is what
this script does; it produces a bounded, sensible decline under stress.

    python forecast.py

Prints the projected cumulative % change per quarter (with a 95% band) and saves
results/adverse_forecast.png.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cleaned_housing_data.csv")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COVARIATES = ["lag_dlog", "mort_rate_30yr", "unrate", "home_sales", "inventory", "neg_equity_share"]

# Adverse macro scenario held fixed over the projection horizon.
ADVERSE = {
    "mort_rate_30yr": 8.0,      # 30-year mortgage rate, %
    "unrate": 6.0,              # unemployment rate, %
    "home_sales": 500.0,        # new home sales (thousands)
    "inventory": 1_200_000.0,   # active listings
    "neg_equity_share": 0.15,   # share of mortgages underwater
}
HORIZON = 8       # quarters
N_SAMPLES = 1000  # posterior (normal-approx) draws
SEED = 42


def load_growth_data(path=DATA):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["region", "date"])
    df["lag_log_hpi"] = df.groupby("region")["log_hpi"].shift(1)
    df["dlog"] = df["log_hpi"] - df["lag_log_hpi"]          # quarterly log growth (stationary)
    df["lag_dlog"] = df.groupby("region")["dlog"].shift(1)  # momentum
    return df.dropna(subset=["dlog", "lag_dlog"])


def main():
    df = load_growth_data()
    scalers = {c: {"mean": df[c].mean(), "std": df[c].std()} for c in COVARIATES}
    dfn = df.copy()
    for c in COVARIATES:
        dfn[c] = (df[c] - scalers[c]["mean"]) / scalers[c]["std"]

    res = smf.mixedlm("dlog ~ " + " + ".join(COVARIATES), dfn, groups=dfn["region"]).fit()
    r2 = 1 - (dfn["dlog"] - res.predict()).var() / dfn["dlog"].var()
    print(f"Growth-model R^2: {r2:.3f}")
    print(res.params[:-1].round(4).to_string())

    # Normal (Laplace) approximation to the posterior of the fixed effects.
    mean = res.params[:-1]
    cov = res.cov_params().iloc[:-1, :-1]
    rng = np.random.default_rng(SEED)
    betas = rng.multivariate_normal(mean.values, cov.values, N_SAMPLES)

    last = df.groupby("region").last().reset_index()
    last_hpi = last.set_index("region")["hpi"]
    last_dlog = last.set_index("region")["dlog"].values
    n_regions = len(last_hpi)

    adv_n = {c: (ADVERSE[c] - scalers[c]["mean"]) / scalers[c]["std"] for c in ADVERSE}
    cum_pct = np.zeros((N_SAMPLES, n_regions, HORIZON))
    for s in range(N_SAMPLES):
        b = betas[s]
        lag_g = last_dlog.copy()
        cum_log = np.zeros(n_regions)
        for t in range(HORIZON):
            lag_g_n = (lag_g - scalers["lag_dlog"]["mean"]) / scalers["lag_dlog"]["std"]
            exog = np.column_stack([
                np.ones(n_regions), lag_g_n,
                np.full(n_regions, adv_n["mort_rate_30yr"]),
                np.full(n_regions, adv_n["unrate"]),
                np.full(n_regions, adv_n["home_sales"]),
                np.full(n_regions, adv_n["inventory"]),
                np.full(n_regions, adv_n["neg_equity_share"]),
            ])
            growth = exog @ b
            cum_log += growth
            cum_pct[s, :, t] = (np.exp(cum_log) - 1) * 100
            lag_g = growth

    national = cum_pct.mean(axis=1)  # average across states
    mean_pct = national.mean(axis=0)
    low = np.quantile(national, 0.025, axis=0)
    high = np.quantile(national, 0.975, axis=0)

    quarters = np.arange(1, HORIZON + 1)
    print("\nAdverse-scenario cumulative % change in HPI:")
    for q, m, lo, hi in zip(quarters, mean_pct, low, high):
        print(f"  Q{q}: {m:6.2f}%   (95% CI [{lo:.2f}, {hi:.2f}])")

    os.makedirs(RESULTS, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(quarters, mean_pct, color="tab:red", marker="o", label="Mean % change")
    plt.fill_between(quarters, low, high, color="tab:red", alpha=0.2, label="95% credible band")
    plt.axhline(0, color="gray", linewidth=1)
    plt.xlabel("Quarters ahead")
    plt.ylabel("Cumulative % change in HPI")
    plt.title("Projected HPI change under an adverse macro scenario")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS, "adverse_forecast.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {os.path.relpath(out, os.path.dirname(os.path.abspath(__file__)))}")


if __name__ == "__main__":
    main()
