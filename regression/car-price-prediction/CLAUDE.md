# CLAUDE.md — car-price-prediction (regression)

Guidance for this specific problem. Repo-wide setup (shared `.venv`, requirements, conventions) lives in the **root** `CLAUDE.md`.

## Problem

Used car selling price regression. Target variable: `selling_price`. Dataset is Indian used-car listings from CarDekho. Self-contained: `datasets/`, `scripts/`, `models/`, and `regression.ipynb` all live in this folder.

The shared environment is at the repo root. For the notebook, select the **"Python (ml)"** kernel. Run commands with the root venv, e.g. `../../.venv/bin/python scripts/preprocessing/run_cleaning.py` (or activate the venv and use plain `python`).

## Running the pipeline

```bash
# from this folder, with the repo-root venv
../../.venv/bin/python scripts/preprocessing/run_cleaning.py
```

This reads `datasets/car-purchase-history/car-details-history-uncleaned.csv` and writes `datasets/car-purchase-history/car-details-history-cleaned.csv`.

## Using the pipeline in code

```python
from scripts.preprocessing.pipeline import build_preprocessing_pipeline, fit_transform_to_dataframe
import pandas as pd

df_raw = pd.read_csv("datasets/car-purchase-history/car-details-history-uncleaned.csv")
pipeline = build_preprocessing_pipeline()
df_clean = fit_transform_to_dataframe(pipeline, df_raw)

X = df_clean.drop(columns=["selling_price"])
y = df_clean["selling_price"]
```

## Architecture

The preprocessing pipeline is a standard sklearn `Pipeline` with three steps, all defined in `scripts/preprocessing/`:

- **`transformers.py`** — two custom `BaseEstimator + TransformerMixin` classes:
  - `RawStringParser` — strips unit strings from `mileage`, `engine`, `max_power`, `torque`, `seats`; handles four torque format variants and two units (Nm / kgm); treats zeroes and implausible values as NaN
  - `FeatureEngineer` — drops all-null spec rows, "Test Drive Car" rows, and duplicate rows; extracts `brand` from `name`; computes `car_age = datetime.now().year - year`; caps `km_driven` at its 99th percentile

- **`pipeline.py`** — assembles the sklearn `Pipeline` and `ColumnTransformer`:
  - Numeric columns → `SimpleImputer(median)`
  - `owner` → `OrdinalEncoder` (First=0 … Fourth+=3)
  - `fuel`, `seller_type`, `transmission`, `brand` → `SimpleImputer(most_frequent)` + `OneHotEncoder`
  - `fit_transform_to_dataframe()` is the main entry point — it handles index alignment between the row-dropping steps and the encoder, and runs a final `drop_duplicates()` on the encoded output

- **`run_cleaning.py`** — CLI entry point, prints a summary and saves the cleaned CSV

Full architecture details and data quality decisions are in `scripts/preprocessing/ARCHITECTURE.md`.

## Dataset stats (post-cleaning)

- Raw: 8,128 rows → Clean: 6,474 rows × 37 columns
- Rows removed: all-null specs (~221), test-drive cars (5), duplicates (1,218)
- `selling_price` range: ₹29,999 – ₹10,000,000 (right-skewed; consider log-transform at modeling stage)
- `car_age` range: 6–32 years (computed dynamically from current year)

## Memory updates

After any major change or addition — new model, new script, new dataset, pipeline modification, key decision — update both memory files:

- **Short-term** (`project-status.md`): update "where we are" and "what's next"
- **Long-term** (`longterm-full-history.md`): append the full detail of what was added, why, and any decisions made

Memory files live at:
`~/.claude/projects/-Users-samratrajsharma-Documents-Personal-Projects-machine-learning-regression/memory/`

Major changes that warrant a memory update:
- A new script or module is created
- A model is trained or evaluated
- A pipeline step is modified
- A data quality issue is found and fixed
- A key design decision is made or reversed

## Key decisions to preserve

- `car_age` uses `datetime.now().year`, not a hardcoded year — keeps ages valid as time passes
- Torque RPM is discarded; only peak torque magnitude (Nm) is kept
- `selling_price` is not log-transformed here — that belongs in the model training script
- Brands with < 20 listings are grouped as `"Other"` in `FeatureEngineer`
