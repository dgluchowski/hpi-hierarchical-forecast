"""Hierarchical (mixed-effects) model of US state-level house prices.

Fits log(HPI) as a function of the previous quarter's HPI and macroeconomic
covariates, with a random intercept per state (region). This characterises which
factors are associated with price *levels* across states and over time.

    python model.py

Note on R^2: with the lagged HPI included the R^2 is very high (~0.96), but that
is largely price momentum (autocorrelation) rather than the macro factors — the
macro covariates alone explain about 0.47. Both numbers are printed below so the
result is not oversold. For the forward-looking adverse-scenario projection see
forecast.py, which models growth (a stationary target) because recursively
forecasting the level model is numerically unstable.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cleaned_housing_data.csv")
COVARIATES = ["lag_hpi", "mort_rate_30yr", "unrate", "home_sales", "inventory", "neg_equity_share"]


def load_data(path=DATA):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.dropna(subset=["lag_hpi"]).sort_values(["region", "date"])


def zscore(df, cols):
    """Standardise covariates (z-score) for numerical stability."""
    out = df.copy()
    for c in cols:
        out[c] = (df[c] - df[c].mean()) / df[c].std()
    return out


def fit_mixed(df, covariates):
    """Fit MixedLM: log_hpi ~ covariates, random intercept per region."""
    dfn = zscore(df, covariates)
    formula = "log_hpi ~ " + " + ".join(covariates)
    return smf.mixedlm(formula, dfn, groups=dfn["region"]).fit()


def r2(res, y):
    resid = y - res.predict()
    return 1 - resid.var() / y.var()


def main():
    df = load_data()
    print(f"{len(df):,} rows | {df['region'].nunique()} regions | "
          f"{df['date'].min().date()} to {df['date'].max().date()}\n")

    # The log_hpi target is not standardised, so df['log_hpi'] matches what the
    # model was fit on (same values and index).
    res = fit_mixed(df, COVARIATES)
    print("Fixed-effect coefficients (standardised covariates):")
    print(res.params[:-1].round(4).to_string())
    print(f"\nR^2 (with lagged HPI):    {r2(res, df['log_hpi']):.4f}")

    res_macro = fit_mixed(df, [c for c in COVARIATES if c != "lag_hpi"])
    print(f"R^2 (macro factors only): {r2(res_macro, df['log_hpi']):.4f}"
          "   <- most of the headline R^2 is the lag, not the macro drivers")


if __name__ == "__main__":
    main()
