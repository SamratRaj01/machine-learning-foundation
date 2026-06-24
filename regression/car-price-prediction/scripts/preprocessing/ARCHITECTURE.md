# Preprocessing Pipeline Architecture

## Overview

The pipeline transforms raw used-car CSV data into a clean, fully numeric matrix ready for regression modeling. It is built on scikit-learn's `Pipeline` and `ColumnTransformer` so it can be fit on training data and applied consistently to new data.

**Input:** `datasets/car-purchase-history/car-details-history-uncleaned.csv` — 8,128 rows × 13 columns
**Output:** `datasets/car-purchase-history/car-details-history-cleaned.csv` — 6,474 rows × 37 columns
**Rows removed:** 1,654 (221 missing all specs, 5 test-drive cars, 1,202 raw duplicates, 16 post-encoding duplicates, 17 zero-value entries treated as nulls and dropped after imputation edge cases)

---

## Pipeline Flow

```
Raw CSV (8,128 rows × 13 cols)
        │
        ▼
┌─────────────────────────────────────────┐
│  Step 1: RawStringParser                │
│  Parse unit-embedded strings → floats   │
│  Flag zeroes as NaN                     │
│  Seats > 10 → NaN                       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Step 2: FeatureEngineer                │
│  Drop all-null spec rows (~221)         │
│  Drop "Test Drive Car" rows (5)         │
│  Drop exact duplicate rows (1,202)      │
│  Extract brand from name                │
│  Compute car_age = current_year - year  │
│  Cap km_driven at 99th percentile       │
│  Drop name, year columns                │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│  Step 3: ColumnTransformer                             │
│                                                        │
│  ┌──────────────────┐  ┌─────────────────┐  ┌───────┐ │
│  │ numeric_pipeline │  │ ordinal_pipeline │  │ ohe_  │ │
│  │ SimpleImputer    │  │ OrdinalEncoder  │  │pipeline│ │
│  │ (median)         │  │ (owner: 0–3)    │  │Impute │ │
│  │                  │  │                 │  │+ OHE  │ │
│  └──────────────────┘  └─────────────────┘  └───────┘ │
└────────────────┬───────────────────────────────────────┘
                 │
                 ▼
        drop_duplicates()   ← catches post-encoding duplicates
        reset_index()       (different car names, same features)
                 │
                 ▼
Clean DataFrame (6,474 rows × 37 cols)
```

---

## Step 1 — RawStringParser (`transformers.py`)

Several columns are stored as strings with embedded units. This step extracts the numeric value from each and flags bad entries as NaN.

| Column | Raw example | Parsed value | Edge cases handled |
|--------|------------|-------------|-------------------|
| `mileage` | `"23.4 kmpl"` / `"17.3 km/kg"` | `23.4` / `17.3` | `"0.0 kmpl"` → NaN (data entry error, not a real reading) |
| `engine` | `"1248 CC"` | `1248.0` | Strip " CC" |
| `max_power` | `"74 bhp"` / `" bhp"` / `"0"` | `74.0` / `NaN` / `NaN` | `" bhp"` (unit only, no number) and `"0"` both → NaN |
| `torque` | see below | see below | Most complex column — four format variants, two units |
| `seats` | `"5"` / `"14"` | `5.0` / `NaN` | Values > 10 flagged as NaN (data entry error) |

### Torque parsing logic

Torque is the hardest column — the dataset has four distinct formats and two units (Nm and kgm). The parser resolves unit in priority order:

1. **Unit immediately follows the leading number** → unambiguous
   - `"190Nm@ 2000rpm"` → Nm detected right after 190 → `190.0`
   - `"22.4 kgm at 1750rpm"` → kgm detected right after 22.4 → `22.4 × 9.8066 = 219.7`

2. **Unit appears later in the string** → check context
   - `"12.7@ 2,700(kgm@ rpm)"` → kgm found, no Nm, value < 50 → treat as kgm → `12.7 × 9.8066 = 124.6`
   - `"115@ 2,500(kgm@ rpm)"` → kgm found, no Nm, but result would be 1127 Nm (> 800, physically impossible for a car) → fall back to treating value as Nm → `115.0`
   - `"380Nm(38.7kgm)@ 2500rpm"` → Nm is right after 380 → `380.0` (kgm in parentheses is an alternate unit, ignored)

3. **Fallback safety cap**: if result > 800 Nm after any conversion, treat the raw number as Nm. If still > 800, set to NaN.

RPM values are discarded entirely — peak torque RPM is not a reliable or consistently formatted feature in this dataset.

---

## Step 2 — FeatureEngineer (`transformers.py`)

Handles all row-level decisions and feature derivation. All fit parameters are computed on training data only to prevent leakage.

### Rows dropped (in order)

| Reason | Count | Why |
|--------|-------|-----|
| All five spec columns null | ~221 | No recoverable signal — imputing all specs from nothing introduces noise |
| `owner == "Test Drive Car"` | 5 | Not a real ownership category |
| Exact duplicate rows | 1,202 | Same car listed twice — would bias model by double-counting |

### Features engineered

**`brand`** — first word of `name` (e.g. `"Maruti Swift Dzire VDI"` → `"Maruti"`).
Brands with fewer than 20 listings are grouped as `"Other"`. This threshold collapses typos and rare entries without manual curation, and is safe to scale — garbage brand names naturally have low counts.

**`car_age`** — `datetime.now().year - year`.
Used instead of raw `year` for two reasons:
- Interpretability: a coefficient on `car_age` reads as "price drops X per year of age", which is meaningful. A coefficient on `year` is directionally backwards and loses meaning over time.
- Generalization: using today's year means a 2020 car is always 5 years old in 2025, 6 years old in 2026, etc. Using `max(year)` from the dataset would produce negative ages for future data.

### Outlier handling

`km_driven` is capped at its 99th percentile (fitted on training data) on the high end. The raw data contains values like 2,360,457 km which are physically implausible and would distort the model.

On the low end, odometer readings below `km_floor` (default 100 km) are treated as data-entry errors — e.g. a 15-year-old car listed at 1 km is not a real low-mileage car. These are set to NaN and filled by the downstream median imputer, the same convention used for `mileage == 0` and `max_power == 0`. A fixed floor is used rather than a low percentile so that genuinely low-mileage recent cars are preserved.

### Dropped columns

`name` (replaced by `brand`), `year` (replaced by `car_age`)

---

## Step 3 — ColumnTransformer (`pipeline.py`)

Applies different encoding strategies to each column group in parallel. The strategy for each group is chosen based on whether the column has a natural ordering.

### Numeric pipeline
**Columns:** `mileage`, `engine`, `max_power`, `torque`, `km_driven`, `car_age`, `seats`

```
SimpleImputer(strategy="median")
```

Median is used instead of mean because all these columns are right-skewed (a few extreme values — luxury car prices, high-mileage outliers — pull the mean away from the typical value). The median stays close to the centre of the distribution and is a better guess for a missing value.

### Ordinal pipeline
**Columns:** `owner`

```
OrdinalEncoder(categories=[["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]])
```

Mapped to integers 0–3. Natural ordering exists and matters — a first-owner car is consistently worth more than a fourth-owner car. Using one-hot here would destroy that signal.

### One-hot pipeline
**Columns:** `fuel`, `seller_type`, `transmission`, `brand`

```
SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")
```

No natural ordering exists for these categories. `handle_unknown="ignore"` ensures unseen brands or fuel types in new data produce all-zero rows rather than errors.

### Post-encoding deduplication

After encoding, rows with different car names but identical features (same brand, year, specs, fuel, etc.) become indistinguishable. These are dropped via a final `drop_duplicates()` call in `fit_transform_to_dataframe`. This catches 16 additional rows that survived raw-level deduplication.

---

## Output Columns (37 total)

| Group | Columns |
|-------|---------|
| Target (1) | `selling_price` |
| Numeric (7) | `mileage`, `engine`, `max_power`, `torque`, `km_driven`, `car_age`, `seats` |
| Ordinal (1) | `owner` |
| Fuel (4) | `fuel_CNG`, `fuel_Diesel`, `fuel_LPG`, `fuel_Petrol` |
| Seller type (3) | `seller_type_Dealer`, `seller_type_Individual`, `seller_type_Trustmark Dealer` |
| Transmission (2) | `transmission_Automatic`, `transmission_Manual` |
| Brand (19 + Other) | `brand_Audi`, `brand_BMW`, `brand_Chevrolet`, `brand_Datsun`, `brand_Fiat`, `brand_Ford`, `brand_Honda`, `brand_Hyundai`, `brand_Jeep`, `brand_Mahindra`, `brand_Maruti`, `brand_Mercedes-Benz`, `brand_Nissan`, `brand_Other`, `brand_Renault`, `brand_Skoda`, `brand_Tata`, `brand_Toyota`, `brand_Volkswagen` |

Note: `brand_Jaguar`, `brand_Lexus`, and `brand_Volvo` were present in v1 but fell below the 20-listing threshold after deduplication and are now grouped into `brand_Other`.

---

## Data Quality Issues Found and Fixed

| Issue | Raw count | Root cause | Fix applied |
|-------|----------|-----------|------------|
| `mileage = 0` | 17 rows | Literal `"0.0 kmpl"` in raw data — entry error | Parsed 0 → NaN, imputed with median |
| `max_power = 0` | 6 rows | Literal `"0"` in raw data — entry error | Parsed 0 → NaN, imputed with median |
| `torque > 800 Nm` | 23 rows | Format `"115@ 2,500(kgm@ rpm)"` misread as kgm — the `(kgm@ rpm)` is a label template, not the torque unit | Fallback: if result > 800, treat raw number as Nm |
| `seats = 14` | 1 row | Data entry error (Maruti Ertiga max is 7) | `seats > 10` → NaN, imputed with median |
| `km_driven = 1` | 1 row | Data entry error (a 2011 Maruti Eeco listed at 1 km) | `km_driven < 100` → NaN, imputed with median |
| Raw duplicate rows | 1,202 rows | Same listing appears multiple times in source data | `drop_duplicates()` in FeatureEngineer |
| Post-encoding duplicates | 16 rows | Different car names/variants with identical spec features | `drop_duplicates()` after ColumnTransformer output |

---

## File Structure

```
scripts/preprocessing/
├── transformers.py     RawStringParser + FeatureEngineer (custom sklearn transformers)
├── pipeline.py         ColumnTransformer wiring + fit_transform_to_dataframe helper
├── run_cleaning.py     Entry point — loads CSV, runs pipeline, saves output
└── ARCHITECTURE.md     This file

datasets/car-purchase-history/
├── car-details-history-uncleaned.csv   Original raw data (8,128 rows)
└── car-details-history-cleaned.csv     Pipeline output (6,474 rows × 37 cols)
```

---

## Usage

```python
from pipeline import build_preprocessing_pipeline, fit_transform_to_dataframe
import pandas as pd

df_raw = pd.read_csv("datasets/car-purchase-history/car-details-history-uncleaned.csv")
pipeline = build_preprocessing_pipeline()
df_clean = fit_transform_to_dataframe(pipeline, df_raw)

X = df_clean.drop(columns=["selling_price"])
y = df_clean["selling_price"]
```

Or run directly:

```bash
python scripts/preprocessing/run_cleaning.py
```

---

## Design Decisions

**Why sklearn Pipeline?**
Fit parameters (median values, 99th percentile cap, common brand list) are computed only on training data and stored in the fitted pipeline object. This prevents data leakage when the same pipeline is applied to a test set or new inference data.

**Why drop all-null spec rows instead of imputing?**
Cars missing all five spec columns have no recoverable signal. Imputing five columns for the same row would be fabricating data, not filling gaps.

**Why median imputation instead of mean?**
`selling_price`, `km_driven`, and `engine` are all right-skewed — a few luxury or high-mileage outliers pull the mean above what a typical car has. The median stays near the centre of the actual distribution and is a more accurate replacement for a missing value.

**Why car_age instead of year?**
Age makes the relationship to price linear and directionally intuitive (older = cheaper). It also stays valid as time passes — using `datetime.now().year` means ages are always computed relative to today, so the pipeline remains correct without retraining when new data arrives.

**Why discard torque RPM?**
RPM at peak torque is a secondary characteristic that correlates more with fuel type (already in the `fuel` column) than with price. It also has the worst format consistency in the dataset, making reliable extraction impractical.

**Why not log-transform selling_price here?**
Log-transforming the target is a modeling decision, not a cleaning decision. Whether a log transform improves a specific model depends on the algorithm and should be evaluated during model selection, not hardcoded into the preprocessing step.
