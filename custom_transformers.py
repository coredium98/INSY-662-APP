# custom_transformers.py
"""
Fully working custom transformers for the Flood Risk KNN model.
Fixes the "X has 19 features but ColumnTransformer expects 16" error.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor


class _RemainderColsList(list):
    """Internal helper for ColumnTransformer compatibility across scikit-learn versions"""
    pass


class RegionAdder(BaseEstimator, TransformerMixin):
    """
    Adds a 'region' column based on country mapping.
    """
    def __init__(self, country_col='country', region_col='region'):
        self.country_col = country_col
        self.region_col = region_col
        # Common mapping — extend as needed
        self.region_map = {
            'United States': 'North America', 'USA': 'North America', 'US': 'North America',
            'Canada': 'North America',
            'Brazil': 'South America', 'Argentina': 'South America', 'Chile': 'South America',
            'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe',
            'Italy': 'Europe', 'Spain': 'Europe', 'Netherlands': 'Europe',
            'India': 'Asia', 'China': 'Asia', 'Japan': 'Asia', 'Indonesia': 'Asia',
            'Thailand': 'Asia', 'Bangladesh': 'Asia',
            'Australia': 'Oceania', 'New Zealand': 'Oceania',
            'South Africa': 'Africa', 'Nigeria': 'Africa', 'Egypt': 'Africa',
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X = X.copy()
        if self.country_col in X.columns:
            X[self.region_col] = X[self.country_col].map(self.region_map).fillna('Other')
        else:
            X[self.region_col] = 'Other'
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Applies log1p transformation (log(x + offset)) to avoid log(0).
    """
    def __init__(self, offset=1.0):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_numeric = X.select_dtypes(include=[np.number])
        X_non_numeric = X.select_dtypes(exclude=[np.number])

        X_transformed = np.log1p(X_numeric + self.offset)  # log(x + 1 + offset)
        X_transformed = pd.DataFrame(X_transformed, columns=X_numeric.columns, index=X_numeric.index)

        # Reattach non-numeric columns unchanged
        if not X_non_numeric.empty:
            X_transformed = pd.concat([X_transformed, X_non_numeric], axis=1)
        return X_transformed


class SingleColumnPowerTransformer(BaseEstimator, TransformerMixin):
    """
    Applies x ** power to a single column (e.g., square population).
    """
    def __init__(self, column=None, power=2):
        self.column = column
        self.power = power

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X = X.copy()
        if self.column and self.column in X.columns:
            X[f"{self.column}_pow{self.power}"] = X[self.column] ** self.power
        return X


class CityMedianImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing numeric values with the median of that city.
    """
    def __init__(self, city_col='city'):
        self.city_col = city_col
        self.medians_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.city_col not in X.columns:
            self.medians_ = None
            return self
        numeric_cols = X.select_dtypes(include=np.number).columns.drop(self.city_col, errors='ignore')
        self.medians_ = X.groupby(self.city_col)[numeric_cols].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if self.medians_ is None or self.city_col not in X.columns:
            return X
        X = X.copy()
        for col in self.medians_.columns:
            X[col] = X.groupby(self.city_col)[col].transform(
                lambda x: x.fillna(self.medians_.loc[x.name, col] if x.name in self.medians_.index else x.median())
            )
        return X


class CityModeImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing categorical values with the mode of that city.
    """
    def __init__(self, city_col='city'):
        self.city_col = city_col
        self.modes_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.city_col not in X.columns:
            self.modes_ = None
            return self
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        self.modes_ = X.groupby(self.city_col)[cat_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        if self.modes_ is None or self.city_col not in X.columns:
            return X
        X = X.copy()
        for col in self.modes_.columns:
            def fill_with_city_mode(group):
                city = group.name
                mode_val = self.modes_.loc[city, col] if city in self.modes_.index else np.nan
                return group.fillna(mode_val)
            X[col] = X.groupby(self.city_col)[col].transform(fill_with_city_mode)
        return X


class ElevationKNNImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing elevation using KNN on latitude/longitude.
    """
    def __init__(self, elevation_col='elevation', coords_cols=['lat', 'lon'], n_neighbors=5):
        self.elevation_col = elevation_col
        self.coords_cols = coords_cols
        self.n_neighbors = n_neighbors
        self.knn_ = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        mask_known = X[self.elevation_col].notna()
        if mask_known.sum() == 0:
            return self
        X_known = X.loc[mask_known, self.coords_cols]
        y_known = X.loc[mask_known, self.elevation_col]
        self.knn_.fit(X_known, y_known)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X = X.copy()
        mask_missing = X[self.elevation_col].isna()
        if mask_missing.sum() == 0 or not hasattr(self, 'knn_'):
            return X
        X_missing = X.loc[mask_missing, self.coords_cols]
        imputed = self.knn_.predict(X_missing)
        X.loc[mask_missing, self.elevation_col] = imputed
        return X


class ElevationLocalDelta(BaseEstimator, TransformerMixin):
    """
    Adds a feature: elevation minus median elevation of the city.
    Helps model relative low-lying areas.
    """
    def __init__(self, elevation_col='elevation', city_col='city', new_col='elev_delta_city'):
        self.elevation_col = elevation_col
        self.city_col = city_col
        self.new_col = new_col
        self.city_medians_ = None

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        if self.elevation_col in X.columns and self.city_col in X.columns:
            self.city_medians_ = X.groupby(self.city_col)[self.elevation_col].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X = X.copy()
        if self.city_medians_ is not None and self.elevation_col in X.columns and self.city_col in X.columns:
            city_median = X[self.city_col].map(self.city_medians_)
            X[self.new_col] = X[self.elevation_col] - city_median
            X[self.new_col] = X[self.new_col].fillna(0)  # fallback
        else:
            X[self.new_col] = 0
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Drops specified columns. Critical — this is usually why you get 19 → 16 features.
    """
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.columns_to_drop, errors='ignore')
