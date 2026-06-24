import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from transformers import RawStringParser, FeatureEngineer

NUMERIC_COLS = ["mileage", "engine", "max_power", "torque", "km_driven", "car_age", "seats"]
ORDINAL_COLS = ["owner"]
ONEHOT_COLS = ["fuel", "seller_type", "transmission", "brand"]

OWNER_ORDER = [["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]]


def build_column_transformer():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    ordinal_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(categories=OWNER_ORDER, handle_unknown="use_encoded_value", unknown_value=4)),
    ])

    onehot_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLS),
            ("ord", ordinal_pipeline, ORDINAL_COLS),
            ("ohe", onehot_pipeline, ONEHOT_COLS),
        ],
        remainder="drop",
    )


def build_preprocessing_pipeline():
    return Pipeline([
        ("parse_strings", RawStringParser()),
        ("feature_engineer", FeatureEngineer()),
        ("encode", build_column_transformer()),
    ])


def get_output_columns(pipeline):
    """Reconstruct column names after ColumnTransformer."""
    ct = pipeline.named_steps["encode"]
    num_names = NUMERIC_COLS
    ord_names = ORDINAL_COLS
    ohe_names = ct.named_transformers_["ohe"]["encoder"].get_feature_names_out(ONEHOT_COLS).tolist()
    return num_names + ord_names + ohe_names


def transform_to_dataframe(pipeline, X_raw, y):
    """Apply an ALREADY-FITTED pipeline to X_raw and reattach selling_price.

    This does NOT fit — it only learns nothing and applies the parameters the
    pipeline already learned. Safe to call on a test/validation/inference split:
    missing values are filled with the *training* median, etc. Call this once per
    split (train and test) after the pipeline has been fit on the training split.

    The row-dropping inside FeatureEngineer means the encoder's output has fewer
    rows than X_raw. We run the fitted steps one at a time so we can capture which
    original-row indices survived and align `y` to them.
    """
    parsed = pipeline.named_steps["parse_strings"].transform(X_raw)
    after_fe = pipeline.named_steps["feature_engineer"].transform(parsed)
    valid_idx = after_fe.index

    X_clean = pipeline.named_steps["encode"].transform(after_fe)
    cols = get_output_columns(pipeline)
    df_clean = pd.DataFrame(X_clean, columns=cols, index=valid_idx)
    df_clean.insert(0, "selling_price", y.loc[valid_idx].values)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    return df_clean


def fit_transform_to_dataframe(pipeline, df_raw):
    """Fit on the full dataset, then clean it.

    Convenience for producing the cleaned CSV for exploration. NOT leakage-safe
    for modeling: it learns medians/caps/categories from every row. When you train
    a model, split first, fit the pipeline on the training split only, then use
    `transform_to_dataframe` on each split. See regression.ipynb.
    """
    X_raw = df_raw.drop(columns=["selling_price"])
    y = df_raw["selling_price"].copy()
    pipeline.fit(X_raw)
    return transform_to_dataframe(pipeline, X_raw, y)
