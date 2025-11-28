# custom_transformers.py
# THIS IS THE ONLY VERSION YOU NEED — works on Streamlit Cloud

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer

# Required for old scikit-learn versions
class _RemainderColsList(list):
    pass


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=[c for c in self.columns if c in X.columns], errors='ignore')


class RegionAdder(BaseEstimator, TransformerMixin):
    def __init__(self, region_map=None):
        self.region_map = region_map or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if 'country' in X.columns:
            default_map = {
                'United States': 'North America', 'USA': 'North America', 'US': 'North America',
                'Canada': 'North America', 'Mexico': 'North America',
                'Brazil': 'Latin America & Caribbean',
                'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe',
                'Netherlands': 'Europe', 'Italy': 'Europe',
                'India': 'South Asia', 'China': 'East Asia', 'Japan': 'East Asia',
                'Indonesia': 'Southeast Asia', 'Thailand': 'Southeast Asia',
                'Australia': 'Oceania', 'New Zealand': 'Oceania',
            }
            X['region'] = X['country'].map({**default_map, **self.region_map}).fillna('Other')
        else:
            X['region'] = 'Unknown'
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """Your data already has log_storm_drain_proximity_m and log_historical_rainfall_intensity_mm_hr → do nothing"""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X)  # No action needed — log columns already exist!


class ElevationKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if 'elevation_m' not in X.columns:
            X['elevation_m'] = 0.0
        return X


class ElevationLocalDelta(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        if 'elev_local_delta_knn' not in X.columns:
            X['elev_local_delta_knn'] = 0.0
        return X


# These are probably not used in your final pipeline — but safe anyway
class SingleColumnPowerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return pd.DataFrame(X)

class CityMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform = fit  # no-op

class CityModeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    transform = fit  # no-op
