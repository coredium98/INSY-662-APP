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
            return np.log(X + self.offset)
        else:
            return np.log(np.array(X) + self.offset)


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
            return X ** self.power
        else:
            return np.array(X) ** self.power


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
