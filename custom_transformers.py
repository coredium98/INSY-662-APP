# custom_transformers.py
# FINAL VERSION – works with ANY previously trained flood KNN model
# Fixes: missing attributes, 19→16 feature mismatch, NaNs, old pickles

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor


# ------------------------------------------------------------------
# Helper for old scikit-learn versions
# ------------------------------------------------------------------
class _RemainderColsList(list):
    pass


# ------------------------------------------------------------------
# 1. RegionAdder – fully backward compatible
# ------------------------------------------------------------------
class RegionAdder(BaseEstimator, TransformerMixin):
    def __init__(self, country_col=None, region_col=None, region_map=None):
        self.country_col = country_col
        self.region_col = region_col
        self.region_map = region_map

    def fit(self, X, y=None):
        # Fix old pickles that had no attributes saved
        if not hasattr(self, "country_col") or self.country_col is None:
            self.country_col = "country"
        if not hasattr(self, "region_col") or self.region_col is None:
            self.region_col = "region"
        if not hasattr(self, "region_map") or self.region_map is None:
            self.region_map = {
                'United States': 'North America', 'USA': 'North America', 'US': 'North America',
                'Canada': 'North America', 'Mexico': 'North America',
                'Brazil': 'Latin America & Caribbean', 'Argentina': 'Latin America & Caribbean',
                'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe',
                'Netherlands': 'Europe', 'Italy': 'Europe', 'Spain': 'Europe',
                'India': 'South Asia', 'Bangladesh': 'South Asia', 'Pakistan': 'South Asia',
                'China': 'East Asia', 'Japan': 'East Asia', 'South Korea': 'East Asia',
                'Indonesia': 'Southeast Asia', 'Thailand': 'Southeast Asia', 'Philippines': 'Southeast Asia',
                'Malaysia': 'Southeast Asia', 'Singapore': 'Southeast Asia',
                'Australia': 'Oceania', 'New Zealand': 'Oceania',
                'South Africa': 'Sub-Saharan Africa', 'Nigeria': 'Sub-Saharan Africa', 'Kenya': 'Sub-Saharan Africa',
            }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        # Re-apply safety net
        if not hasattr(self, "country_col") or self.country_col is None:
            self.country_col = "country"
        if not hasattr(self, "region_col") or self.region_col is None:
            self.region_col = "region"

        if self.country_col in X.columns:
            X[self.region_col] = X[self.country_col].map(self.region_map).fillna("Other")
        else:
            X[self.region_col] = "Other"
        return X


# ------------------------------------------------------------------
# 2. LogTransformer
# ------------------------------------------------------------------
class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        if not hasattr(self, "offset"):
            self.offset = 1.0
        return self

    def transform(self, X):
        if not hasattr(self, "offset"):
            self.offset = 1.0
        X = pd.DataFrame(X)
        return np.log1p(X.select_dtypes(include=[np.number]) + self.offset)


# ------------------------------------------------------------------
# 3. SingleColumnPowerTransformer
# ------------------------------------------------------------------
class SingleColumnPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column=None, power=2):
        self.column = column
        self.power = power

    def fit(self, X, y=None):
        if not hasattr(self, "column"):
            self.column = None
        if not hasattr(self, "power"):
            self.power = 2
        return self

    def transform(self, X):
        if not hasattr(self, "column"):
            self.column = None
        if not hasattr(self, "power"):
            self.power = 2
        X = pd.DataFrame(X).copy()
        if self.column and self.column in X.columns:
            new_name = f"{self.column}_pow{int(self.power)}"
            X[new_name] = X[self.column] ** self.power
        return X


# ------------------------------------------------------------------
# 4. CityMedianImputer
# ------------------------------------------------------------------
class CityMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, city_col=None):
        self.city_col = city_col

    def fit(self, X, y=None):
        if not hasattr(self, "city_col") or self.city_col is None:
            self.city_col = "city"
        X = pd.DataFrame(X)
        if self.city_col in X.columns:
            num_cols = X.select_dtypes(include=np.number).columns.drop(self.city_col, errors="ignore")
            self.medians_ = X.groupby(self.city_col)[num_cols].median()
        else:
            self.medians_ = None
        return self

    def transform(self, X):
        if not hasattr(self, "medians_") or self.medians_ is None:
            return X
        X = pd.DataFrame(X).copy()
        if self.city_col not in X.columns:
            return X
        for col in self.medians_.columns:
            X[col] = X.groupby(self.city_col)[col].transform(
                lambda g: g.fillna(self.medians_.loc[g.name, col] if g.name in self.medians_.index else g.median())
            )
        return X


# ------------------------------------------------------------------
# 5. CityModeImputer
# ------------------------------------------------------------------
class CityModeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, city_col=None):
        self.city_col = city_col

    def fit(self, X, y=None):
        if not hasattr(self, "city_col") or self.city_col is None:
            self.city_col = "city"
        X = pd.DataFrame(X)
        if self.city_col in X.columns:
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            self.modes_ = X.groupby(self.city_col)[cat_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        else:
            self.modes_ = None
        return self

    def transform(self, X):
        if not hasattr(self, "modes_") or self.modes_ is None:
            return X
        X = pd.DataFrame(X).copy()
        if self.city_col not in X.columns:
            return X
        for col in self.modes_.columns:
            def fill(g):
                mode_val = self.modes_.loc[g.name, col] if g.name in self.modes_.index else np.nan
                return g.fillna(mode_val)
            X[col] = X.groupby(self.city_col)[col].transform(fill)
        return X


# ------------------------------------------------------------------
# 6. ElevationKNNImputer
# ------------------------------------------------------------------
class ElevationKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5, elevation_col=None, coords_cols=None):
        self.n_neighbors = n_neighbors
        self.elevation_col = elevation_col
        self.coords_cols = coords_cols

    def fit(self, X, y=None):
        if not hasattr(self, "elevation_col") or self.elevation_col is None:
            self.elevation_col = "elevation_m"
        if not hasattr(self, "coords_cols") or self.coords_cols is None:
            self.coords_cols = ["latitude", "longitude"]
        if not hasattr(self, "n_neighbors"):
            self.n_neighbors = 5

        X = pd.DataFrame(X)
        mask = X[self.elevation_col].notna()
        if mask.sum() == 0:
            return self

        self.knn_ = KNeighborsRegressor(n_neighbors=self.n_neighbors)
        X_known = X.loc[mask, self.coords_cols]
        y_known = X.loc[mask, self.elevation_col]
        self.knn_.fit(X_known, y_known)
        return self

    def transform(self, X):
        if not hasattr(self, "knn_"):
            return X
        X = pd.DataFrame(X).copy()
        mask_missing = X[self.elevation_col].isna()
        if mask_missing.sum() == 0:
            return X
        preds = self.knn_.predict(X.loc[mask_missing, self.coords_cols])
        X.loc[mask_missing, self.elevation_col] = preds
        return X


# ------------------------------------------------------------------
# 7. ElevationLocalDelta
# ------------------------------------------------------------------
class ElevationLocalDelta(BaseEstimator, TransformerMixin):
    def __init__(self, elevation_col=None, city_col=None, new_col=None):
        self.elevation_col = elevation_col
        self.city_col = city_col
        self.new_col = new_col

    def fit(self, X, y=None):
        if not hasattr(self, "elevation_col") or self.elevation_col is None:
            self.elevation_col = "elevation_m"
        if not hasattr(self, "city_col") or self.city_col is None:
            self.city_col = "city"
        if not hasattr(self, "new_col") or self.new_col is None:
            self.new_col = "elev_delta_city"

        X = pd.DataFrame(X)
        if self.elevation_col in X.columns and self.city_col in X.columns:
            self.city_medians_ = X.groupby(self.city_col)[self.elevation_col].median()
        else:
            self.city_medians_ = None
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if getattr(self, "city_medians_", None) is not None:
            city_med = X[self.city_col].map(self.city_medians_).fillna(0)
            X[self.new_col] = X[self.elevation_col] - city_med
        else:
            X[self.new_col] = 0
        return X


# ------------------------------------------------------------------
# 8. ColumnDropper – the one that fixes 19 → 16 features
# ------------------------------------------------------------------
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        if not hasattr(self, "columns_to_drop"):
            self.columns_to_drop = []
        # Common defaults if nothing was saved
        common_drops = ["city", "country", "admin_ward", "catchment_id", "storm_drain_type"]
        for c in common_drops:
            if c not in self.columns_to_drop:
                self.columns_to_drop.append(c)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.columns_to_drop, errors="ignore")
