"""Build the cleaned quarterly panel from the raw source files.

Merges seven public/economic sources into one quarterly, state-level dataset:
regional FHFA HPI plus national mortgage rates, unemployment, home sales,
inventory, and negative-equity share. Produces ``data/cleaned_housing_data.csv``
(already included in this repo).

The raw source files are **not** committed (see SOURCES.md for links). To
regenerate from scratch, download them into ``data/raw/`` and run:

    python preprocessing.py
"""

import os

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cleaned_housing_data.csv")


def _q(path):
    return os.path.join(RAW, path)


def main():
    actlis = pd.read_csv(_q("ACTLISCOUUS.csv"))
    csus = pd.read_csv(_q("CSUSHPINSA.csv"))
    pmms = pd.read_csv(_q("Freddie Mac Primary Mortgage Market Survey (PMMS).csv"), header=0)
    hpi = pd.read_csv(_q("hpi_master.csv"))
    hsn = pd.read_csv(_q("HSN1F.csv"))
    neg_eq = pd.read_csv(_q("NegativeEquity_2017Q1_Public.csv"))
    unrate = pd.read_csv(_q("UNRATE.csv"))

    # --- National series: monthly/weekly -> quarter-end means, 2000+ ---
    actlis["date"] = pd.to_datetime(actlis["observation_date"])
    actlis = (actlis[actlis["date"] >= "2000-01-01"].rename(columns={"ACTLISCOUUS": "inventory"})
              .drop("observation_date", axis=1).set_index("date").resample("QE").mean())

    csus["date"] = pd.to_datetime(csus["observation_date"])
    csus = (csus[csus["date"] >= "2000-01-01"].rename(columns={"CSUSHPINSA": "national_hpi"})
            .drop("observation_date", axis=1).set_index("date").resample("QE").mean())

    pmms = pmms.rename(columns={"Week": "date", "FRM": "mort_rate_30yr"})
    pmms["date"] = pd.to_datetime(pmms["date"], errors="coerce")
    pmms["mort_rate_30yr"] = pd.to_numeric(pmms["mort_rate_30yr"], errors="coerce")
    pmms = (pmms[pmms["date"] >= "2000-01-01"].set_index("date")
            .resample("QE").mean(numeric_only=True)["mort_rate_30yr"].to_frame())

    hsn["date"] = pd.to_datetime(hsn["observation_date"])
    hsn = (hsn[hsn["date"] >= "2000-01-01"].rename(columns={"HSN1F": "home_sales"})
           .drop("observation_date", axis=1).set_index("date").resample("QE").mean())

    unrate["date"] = pd.to_datetime(unrate["observation_date"])
    unrate = (unrate[unrate["date"] >= "2000-01-01"].rename(columns={"UNRATE": "unrate"})
              .drop("observation_date", axis=1).set_index("date").resample("QE").mean())

    # --- Regional FHFA HPI: state-level, purchase-only, quarterly ---
    hpi = hpi[(hpi["hpi_type"] == "traditional") & (hpi["hpi_flavor"] == "purchase-only")
              & (hpi["frequency"] == "quarterly") & (hpi["level"] == "State")]
    hpi["date"] = pd.to_datetime(hpi["yr"].astype(str) + "-"
                                 + ((hpi["period"] * 3) - 2).astype(str).str.zfill(2) + "-01")
    hpi["date"] = hpi["date"] + pd.offsets.QuarterEnd(0)
    hpi = (hpi[hpi["date"] >= "2000-01-01"].rename(columns={"place_name": "region", "index_nsa": "hpi"}))
    hpi_long = hpi[["date", "region", "hpi"]].dropna(subset=["hpi"])

    # --- Regional negative-equity share (2011-2017), long format ---
    neg_eq = neg_eq[neg_eq["RegionType"] == "State"]
    neg_eq = neg_eq.melt(id_vars=["RegionName"],
                         value_vars=[c for c in neg_eq.columns if "Q" in c],
                         var_name="quarter", value_name="neg_equity_share")
    neg_eq["yr"] = neg_eq["quarter"].str[:4].astype(int)
    neg_eq["qtr"] = neg_eq["quarter"].str[5:].astype(int)
    neg_eq["date"] = pd.to_datetime(neg_eq["yr"].astype(str) + "-"
                                    + ((neg_eq["qtr"] * 3) - 2).astype(str).str.zfill(2) + "-01")
    neg_eq["date"] = neg_eq["date"] + pd.offsets.QuarterEnd(0)
    neg_eq_long = (neg_eq.rename(columns={"RegionName": "region"})
                   [["date", "region", "neg_equity_share"]])

    # --- Merge: regional HPI <- national indicators <- regional neg-equity ---
    national = pd.concat([csus, pmms, unrate, hsn, actlis], axis=1).reset_index()
    national["date"] = pd.to_datetime(national["date"])
    df = hpi_long.merge(national, on="date", how="left")
    df = df.merge(neg_eq_long, on=["date", "region"], how="left")

    # --- Missing values / outliers, feature engineering ---
    df = df.set_index(["date", "region"]).ffill().bfill()
    df = df.clip(lower=df.quantile(0.01), upper=df.quantile(0.99), axis=1).reset_index()
    df["log_hpi"] = np.log(df["hpi"] + 1e-6)
    df["lag_hpi"] = df.groupby("region")["hpi"].shift(1)
    df["hpi_change"] = df.groupby("region")["hpi"].pct_change()

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(df):,} rows, {df['region'].nunique()} regions")


if __name__ == "__main__":
    main()
