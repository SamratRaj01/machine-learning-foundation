import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import build_preprocessing_pipeline, fit_transform_to_dataframe

RAW_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../datasets/car-purchase-history/car-details-history-uncleaned.csv",
)
OUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../datasets/car-purchase-history/car-details-history-cleaned.csv",
)


def main():
    print("Loading raw data...")
    df_raw = pd.read_csv(RAW_PATH)
    print(f"  Raw rows: {len(df_raw)}")

    print("Building and running pipeline...")
    pipeline = build_preprocessing_pipeline()
    df_clean = fit_transform_to_dataframe(pipeline, df_raw)

    print(f"  Clean rows: {len(df_clean)}")
    print(f"  Columns ({len(df_clean.columns)}): {list(df_clean.columns)}")

    null_counts = df_clean.isnull().sum()
    if null_counts.any():
        print("\nWARNING — nulls remain:")
        print(null_counts[null_counts > 0])
    else:
        print("  No nulls remaining.")

    print("\nSummary stats:")
    print(df_clean[["selling_price", "km_driven", "engine", "max_power", "car_age"]].describe().round(1))

    df_clean.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
