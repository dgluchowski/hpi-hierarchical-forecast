"""Panel I/O tests.

The derived file ``data/cleaned_housing_data.csv`` is not part of the
repository. Tests that need a panel use a small synthetic fixture. The check
against a rebuilt file skips when that file is absent.
"""

import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model
import preprocessing
from forecast import load_growth_data

PANEL_COLUMNS = [
    "date",
    "region",
    "hpi",
    "national_hpi",
    "mort_rate_30yr",
    "unrate",
    "home_sales",
    "inventory",
    "neg_equity_share",
    "log_hpi",
    "lag_hpi",
    "hpi_change",
]


def _write_raw(raw):
    """Minimal raw files with the column names preprocessing.py reads."""
    months = pd.date_range("2000-01-01", "2000-12-01", freq="MS")
    for name, col in (
        ("ACTLISCOUUS.csv", "ACTLISCOUUS"),
        ("CSUSHPINSA.csv", "CSUSHPINSA"),
        ("HSN1F.csv", "HSN1F"),
        ("UNRATE.csv", "UNRATE"),
    ):
        pd.DataFrame(
            {"observation_date": months.strftime("%Y-%m-%d"), col: range(100, 100 + len(months))}
        ).to_csv(os.path.join(raw, name), index=False)

    weeks = pd.date_range("2000-01-06", "2000-12-28", freq="W-THU")
    pd.DataFrame(
        {"Week": weeks.strftime("%Y-%m-%d"), "FRM": [6.5 + 0.02 * i for i in range(len(weeks))]}
    ).to_csv(
        os.path.join(raw, "Freddie Mac Primary Mortgage Market Survey (PMMS).csv"),
        index=False,
    )

    rows = []
    for region in ("Alaska", "Alabama"):
        for period in (1, 2, 3, 4):
            rows.append(
                {
                    "hpi_type": "traditional",
                    "hpi_flavor": "purchase-only",
                    "frequency": "quarterly",
                    "level": "State",
                    "yr": 2000,
                    "period": period,
                    "place_name": region,
                    "index_nsa": 100 + period,
                }
            )
    # Dropped by the state / purchase-only / quarterly filter.
    rows.append(
        {
            "hpi_type": "traditional",
            "hpi_flavor": "all-transactions",
            "frequency": "monthly",
            "level": "USA",
            "yr": 2000,
            "period": 1,
            "place_name": "United States",
            "index_nsa": 50,
        }
    )
    pd.DataFrame(rows).to_csv(os.path.join(raw, "hpi_master.csv"), index=False)

    neg = pd.DataFrame(
        {
            "RegionType": ["State", "State", "Metro"],
            "RegionName": ["Alaska", "Alabama", "Anchorage"],
            "2000Q1": [0.10, 0.11, 0.90],
            "2000Q2": [0.12, 0.13, 0.91],
            "2000Q3": [0.14, 0.15, 0.92],
            "2000Q4": [0.16, 0.17, 0.93],
        }
    )
    neg.to_csv(os.path.join(raw, "NegativeEquity_2017Q1_Public.csv"), index=False)


class SyntheticPanelTests(unittest.TestCase):
    def test_preprocessing_and_loaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = os.path.join(tmp, "raw")
            os.makedirs(raw)
            _write_raw(raw)
            out = os.path.join(tmp, "cleaned_housing_data.csv")
            orig_raw, orig_out = preprocessing.RAW, preprocessing.OUT
            preprocessing.RAW, preprocessing.OUT = raw, out
            try:
                preprocessing.main()
            finally:
                preprocessing.RAW, preprocessing.OUT = orig_raw, orig_out

            built = pd.read_csv(out)
            for col in PANEL_COLUMNS:
                self.assertIn(col, built.columns)
            self.assertEqual(set(built["region"]), {"Alaska", "Alabama"})
            self.assertFalse(built["hpi"].isna().any())

            loaded = model.load_data(out)
            self.assertFalse(loaded["lag_hpi"].isna().any())
            self.assertGreater(len(loaded), 0)

            scored = model.zscore(loaded, model.COVARIATES)
            for col in model.COVARIATES:
                self.assertAlmostEqual(scored[col].mean(), 0.0, places=6)

            growth = load_growth_data(out)
            self.assertGreater(len(growth), 0)
            self.assertIn("dlog", growth.columns)
            self.assertFalse(growth["lag_dlog"].isna().any())


class RebuiltPanelTests(unittest.TestCase):
    def test_rebuilt_csv_has_panel_columns(self):
        if not os.path.exists(model.DATA):
            self.skipTest(
                "data/cleaned_housing_data.csv is absent; rebuild it with preprocessing.py"
            )
        df = pd.read_csv(model.DATA)
        for col in PANEL_COLUMNS:
            self.assertIn(col, df.columns)
        self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
