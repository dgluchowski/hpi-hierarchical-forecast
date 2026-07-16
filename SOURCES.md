# Data Sources

The analysis-ready panel (`data/cleaned_housing_data.csv`) is derived from the
following public sources. Raw files are not committed; download them into
`data/raw/` and run `preprocessing.py` to rebuild.

| Series | Source |
|---|---|
| FHFA House Price Index (state, purchase-only, quarterly) | https://www.fhfa.gov/data/hpi/datasets |
| 30-year mortgage rate (Freddie Mac PMMS) | https://www.freddiemac.com/pmms |
| New one-family house sales (HSN1F) | https://fred.stlouisfed.org/series/HSN1F |
| Active listing count (ACTLISCOUUS) | https://fred.stlouisfed.org/series/ACTLISCOUUS |
| Case-Shiller national HPI (CSUSHPINSA) | https://fred.stlouisfed.org/series/CSUSHPINSA |
| Unemployment rate (UNRATE) | https://fred.stlouisfed.org/series/UNRATE |
| US housing market factors (Kaggle) | https://www.kaggle.com/datasets/faryarmemon/usa-housing-market-factors |
| Negative-equity trends (Kaggle) | https://www.kaggle.com/datasets/thedevastator/negative-equity-trends-in-us-housing-markets-201 |

FHFA, Freddie Mac, and FRED data are public. The two Kaggle datasets are subject to
their own licenses on Kaggle; please review them before redistributing raw rows.
