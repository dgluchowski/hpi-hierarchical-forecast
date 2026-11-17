# Data Sources

`preprocessing.py` builds `data/cleaned_housing_data.csv` from the raw files
below. Neither those files nor the derived panel are committed. Download each
file into `data/raw/` under the name in the table, then run:

```bash
python preprocessing.py
```

The README lists the columns each file must have. `model.py` and `forecast.py`
read the rebuilt CSV. There are no notebooks in this repository.

| Series | File in `data/raw/` | Source |
|---|---|---|
| FHFA House Price Index (state, purchase-only, quarterly) | `hpi_master.csv` | https://www.fhfa.gov/data/hpi/datasets |
| 30-year mortgage rate (Freddie Mac PMMS) | `Freddie Mac Primary Mortgage Market Survey (PMMS).csv` | https://www.freddiemac.com/pmms |
| New one-family house sales (HSN1F, Census) | `HSN1F.csv` | https://fred.stlouisfed.org/series/HSN1F |
| Unemployment rate (UNRATE, BLS) | `UNRATE.csv` | https://fred.stlouisfed.org/series/UNRATE |
| Active listing count (ACTLISCOUUS, Realtor.com inventory) | `ACTLISCOUUS.csv` | https://fred.stlouisfed.org/series/ACTLISCOUUS |
| Case-Shiller national HPI (CSUSHPINSA, S&P Cotality) | `CSUSHPINSA.csv` | https://fred.stlouisfed.org/series/CSUSHPINSA |
| Negative-equity share (Zillow, via Kaggle) | `NegativeEquity_2017Q1_Public.csv` | https://www.kaggle.com/datasets/thedevastator/negative-equity-trends-in-us-housing-markets-201 |

FHFA HPI (US government work; Source: Federal Housing Finance Agency, FHFA HPI®) and Census/BLS series are public. Freddie Mac PMMS is used with attribution (© Freddie Mac). Case-Shiller (S&P Cotality), Realtor.com inventory, and Zillow negative-equity data are third-party and remain under their owners' terms; they are not redistributed here and are not covered by the MIT license. Zillow data © Zillow Group, via Kaggle.
