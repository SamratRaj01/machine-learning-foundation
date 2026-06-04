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


def fit_transform_to_dataframe(pipeline, df_raw):
    """Fit pipeline and return a clean DataFrame with named columns."""
    X_raw = df_raw.drop(columns=["selling_price"])
    y = df_raw["selling_price"].copy()

    # Align index: drop rows removed by FeatureEngineer before fitting
    fe = FeatureEngineer()
    after_parse = RawStringParser().fit_transform(X_raw)
    after_fe = fe.fit(after_parse).transform(after_parse)
    valid_idx = after_fe.index

    pipeline.fit(X_raw.loc[valid_idx])
    X_clean = pipeline.transform(X_raw.loc[valid_idx])

    cols = get_output_columns(pipeline)
    df_clean = pd.DataFrame(X_clean, columns=cols, index=valid_idx)
    df_clean.insert(0, "selling_price", y.loc[valid_idx].values)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    return df_clean
