import re
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class RawStringParser(BaseEstimator, TransformerMixin):
    """Converts string columns with embedded units into numeric values."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["mileage"] = df["mileage"].apply(self._parse_mileage)
        df["engine"] = df["engine"].apply(self._parse_engine)
        df["max_power"] = df["max_power"].apply(self._parse_max_power)
        df["torque"] = df["torque"].apply(self._parse_torque)
        df["seats"] = pd.to_numeric(df["seats"], errors="coerce").apply(
            lambda v: np.nan if pd.notna(v) and v > 10 else v
        )
        return df

    def _parse_mileage(self, val):
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        match = re.search(r"[\d.]+", str(val))
        result = float(match.group()) if match else np.nan
        return np.nan if result == 0 else result

    def _parse_engine(self, val):
        if pd.isna(val) or str(val).strip() == "":
            return np.nan
        match = re.search(r"[\d.]+", str(val))
        return float(match.group()) if match else np.nan

    def _parse_max_power(self, val):
        if pd.isna(val) or str(val).strip() in ("", "bhp"):
            return np.nan
        match = re.search(r"[\d.]+", str(val))
        result = float(match.group()) if match else np.nan
        return np.nan if result == 0 else result

    def _parse_torque(self, val):
        """
        Extracts torque in Nm. Format variants handled:
          - "190Nm@ 2000rpm"           → 190.0  (Nm explicit after leading number)
          - "22.4 kgm at 1750rpm"      → 219.7  (kgm explicit after leading number)
          - "12.7@ 2,700(kgm@ rpm)"   → 124.6  (small value, kgm unit label)
          - "115@ 2,500(kgm@ rpm)"    → 115.0  (kgm is format label, not torque unit)
          - "380Nm(38.7kgm)@ 2500rpm" → 380.0  (Nm explicit, kgm is alternate unit)
        """
        if pd.isna(val) or str(val).strip() == "":
            return np.nan

        val_str = str(val).strip()
        leading = re.match(r"^[\d.,]+", val_str)
        if not leading:
            return np.nan

        numeric = float(leading.group().replace(",", ""))
        after = val_str[len(leading.group()):]

        # Unit immediately follows the leading number → unambiguous
        if re.match(r"\s*[Nn][Mm]", after):
            return numeric
        if re.match(r"\s*kgm", after, re.IGNORECASE):
            return round(numeric * 9.8066, 2)

        # kgm appears later: check if it's a format label "(kgm@ rpm)" vs real unit
        has_kgm = bool(re.search(r"kgm", val_str, re.IGNORECASE))
        has_nm = bool(re.search(r"[Nn][Mm]", val_str))
        result = round(numeric * 9.8066, 2) if (has_kgm and not has_nm) or (not has_nm and numeric < 50) else numeric

        # If result is physically implausible (> 800 Nm), the kgm conversion was wrong
        if result > 800:
            result = numeric
        return result if result > 0 else np.nan


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Drops bad rows, engineers brand and car_age, caps km_driven outliers."""

    def __init__(self, rare_brand_threshold=20, km_cap_percentile=99):
        self.rare_brand_threshold = rare_brand_threshold
        self.km_cap_percentile = km_cap_percentile
        self._km_cap = None
        self._common_brands = None

    def fit(self, X, y=None):
        df = X.copy()
        df = self._drop_bad_rows(df)
        self._km_cap = np.percentile(df["km_driven"].dropna(), self.km_cap_percentile)
        brands = df["name"].str.split().str[0]
        counts = brands.value_counts()
        self._common_brands = set(counts[counts >= self.rare_brand_threshold].index)
        return self

    def transform(self, X):
        df = X.copy()
        df = self._drop_bad_rows(df)

        df["brand"] = df["name"].str.split().str[0]
        df["brand"] = df["brand"].apply(
            lambda b: b if b in self._common_brands else "Other"
        )

        df["car_age"] = datetime.now().year - df["year"].astype(int)
        df["km_driven"] = df["km_driven"].clip(upper=self._km_cap)

        df = df.drop(columns=["name", "year"])
        return df

    def _drop_bad_rows(self, df):
        spec_cols = ["mileage", "engine", "max_power", "torque", "seats"]
        all_null = df[spec_cols].isnull().all(axis=1)
        df = df[~all_null].copy()
        df = df[df["owner"] != "Test Drive Car"].copy()
        df = df.drop_duplicates().copy()
        return df
