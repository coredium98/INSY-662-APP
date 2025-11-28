# custom_transformers.py — FINAL BULLETPROOF VERSION
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

def to_df(X):
    if isinstance(X, pd.DataFrame):
        return X.copy()
    try:
        return pd.DataFrame(X)
    except:
        return pd.DataFrame(np.array(X))

class _RemainderColsList(list): pass

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None): self.columns = columns or []
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = to_df(X)
        return X.drop(columns=[c for c in self.columns if c in X.columns], errors='ignore')

class RegionAdder(BaseEstimator, TransformerMixin):
    def __init__(self, region_map=None): self.region_map = region_map or {}
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = to_df(X)
        if 'country' in X.columns:
            X['region'] = X['country'].map(self.region_map).fillna('Other')
        else:
            X['region'] = 'Unknown'
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return to_df(X)

class ElevationKNNImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = to_df(X)
        if 'elevation_m' not in X.columns:
            X['elevation_m'] = 0.0
        return X

class ElevationLocalDelta(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = to_df(X)
        if 'elev_local_delta_knn' not in X.columns:
            X['elev_local_delta_knn'] = 0.0
        return X

# Safe no-ops
class SingleColumnPowerTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return to_df(X)

class CityMedianImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    transform = fit

class CityModeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    transform = fit
