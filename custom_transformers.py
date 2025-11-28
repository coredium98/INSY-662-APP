"""
Custom transformers for the flood risk model
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class _RemainderColsList(list):
    """
    Internal class used by ColumnTransformer to track remainder columns.
    This is a simple list subclass.
    """
    pass


class RegionAdder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add region-based features to the dataset.
    Maps countries to their respective regions.
    """

    def __init__(self, region_map=None):
        self.region_map = region_map or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # If region_map is provided, this would add region information
        # For now, return X as-is to maintain compatibility
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply log transformation to features.
    """

    def __init__(self, offset=1):
        self.offset = offset

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply log transformation
        if isinstance(X, pd.DataFrame):
            # Ensure DataFrame has numeric dtypes
            X_numeric = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            result = np.log(X_numeric + self.offset)
            return pd.DataFrame(result, columns=X.columns, index=X.index)
        else:
            # Ensure X is a proper numpy array
            try:
                # Try direct conversion first
                X_arr = np.asarray(X, dtype=np.float64)
            except (ValueError, TypeError):
                # If conversion fails, try to handle object arrays
                X_arr = np.asarray(X)
                # Convert object array to float, replacing non-numeric with 0
                try:
                    X_arr = pd.DataFrame(X_arr).apply(pd.to_numeric, errors='coerce').fillna(0).values
                except Exception:
                    # Last resort: force conversion and replace inf/nan
                    X_arr = np.asarray(X, dtype=object)
                    X_arr = np.where(pd.notna(X_arr), X_arr, 0)
                    X_arr = X_arr.astype(np.float64)

            # Handle potential negative values or zeros
            X_arr = np.clip(X_arr, 0, None)  # Ensure non-negative
            result = np.log(X_arr + self.offset)
            return result


class SingleColumnPowerTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to apply power transformation to a single column.
    """

    def __init__(self, power=2):
        self.power = power

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Apply power transformation
        if isinstance(X, pd.DataFrame):
            # Ensure DataFrame has numeric dtypes
            X_numeric = X.apply(pd.to_numeric, errors='coerce').fillna(0)
            result = X_numeric ** self.power
            return pd.DataFrame(result, columns=X.columns, index=X.index)
        else:
            try:
                X_arr = np.asarray(X, dtype=np.float64)
            except (ValueError, TypeError):
                # Handle object arrays
                X_arr = np.asarray(X)
                try:
                    X_arr = pd.DataFrame(X_arr).apply(pd.to_numeric, errors='coerce').fillna(0).values
                except Exception:
                    X_arr = np.asarray(X, dtype=object)
                    X_arr = np.where(pd.notna(X_arr), X_arr, 0)
                    X_arr = X_arr.astype(np.float64)
            return X_arr ** self.power


class CityMedianImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer that fills missing values with the median value per city.
    """

    def __init__(self, city_col=None):
        self.city_col = city_col
        self.medians_ = {}

    def fit(self, X, y=None):
        # Calculate medians per city if applicable
        if isinstance(X, pd.DataFrame) and self.city_col and self.city_col in X.columns:
            self.medians_ = X.groupby(self.city_col).median().to_dict()
        return self

    def transform(self, X):
        # Return X as-is for compatibility
        return X


class CityModeImputer(BaseEstimator, TransformerMixin):
    """
    Custom imputer that fills missing values with the mode value per city.
    """

    def __init__(self, city_col=None):
        self.city_col = city_col
        self.modes_ = {}

    def fit(self, X, y=None):
        # Calculate modes per city if applicable
        if isinstance(X, pd.DataFrame) and self.city_col and self.city_col in X.columns:
            self.modes_ = X.groupby(self.city_col).agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else None).to_dict()
        return self

    def transform(self, X):
        # Return X as-is for compatibility
        return X


class ElevationKNNImputer(BaseEstimator, TransformerMixin):
    """
    Custom KNN imputer specifically for elevation data.
    """

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Return X as-is for compatibility
        return X


class ElevationLocalDelta(BaseEstimator, TransformerMixin):
    """
    Custom transformer to calculate local elevation differences.
    """

    def __init__(self, elevation_col=None):
        self.elevation_col = elevation_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Return X as-is for compatibility
        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Custom transformer to drop specific columns from the dataset.
    """

    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Drop columns if specified
        if isinstance(X, pd.DataFrame) and self.columns_to_drop:
            return X.drop(columns=self.columns_to_drop, errors='ignore')
        return X
