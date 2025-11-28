# custom_transformers.py
# EXACT MATCH to your notebook's custom transformers
# With full backward compatibility for pickled models

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.neighbors import NearestNeighbors


# Helper for old scikit-learn versions
class _RemainderColsList(list):
    pass


# In custom_transformers.py — replace the entire RegionAdder class
class RegionAdder(BaseEstimator, TransformerMixin):
    def __init__(self, region_map=None):
        self.region_map = region_map or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        
        # If 'country' column is missing, just add a dummy 'region' = 'Unknown'
        if 'country' not in X.columns:
            X['region'] = 'Unknown'
            return X
            
        # Otherwise do the mapping
        default_map = {
            'United States': 'North America', 'USA': 'North America',
            'Canada': 'North America', 'Mexico': 'North America',
            'Brazil': 'Latin America & Caribbean',
            'United Kingdom': 'Europe', 'Germany': 'Europe', 'France': 'Europe',
            'Netherlands': 'Europe', 'Italy': 'Europe',
            'India': 'South Asia', 'China': 'East Asia', 'Japan': 'East Asia',
            'Indonesia': 'Southeast Asia', 'Thailand': 'Southeast Asia',
            'Australia': 'Oceania', 'New Zealand': 'Oceania',
            'Nigeria': 'Sub-Saharan Africa', 'Kenya': 'Sub-Saharan Africa',
        }
        full_map = {**default_map, **self.region_map}
        X['region'] = X['country'].map(full_map).fillna('Other')
        return X

class LogTransformer(BaseEstimator, TransformerMixin):
    """Create log-transformed versions of specified positive columns."""
    def __init__(self, cols=None, new_cols=None):
        self.cols = cols
        self.new_cols = new_cols

    def fit(self, X, y=None):
        if not hasattr(self, "cols") or self.cols is None:
            self.cols = []
        if not hasattr(self, "new_cols") or self.new_cols is None:
            self.new_cols = []
        return self

    def transform(self, X):
        X = X.copy()
        for col, new_col in zip(self.cols, self.new_cols):
            if new_col not in X.columns:
                # small constant avoids log(0)
                X[new_col] = np.log(X[col] + 1e-6)
        return X


class SingleColumnPowerTransformer(BaseEstimator, TransformerMixin):
    """Apply Yeo-Johnson to a single column."""
    def __init__(self, column=None):
        self.column = column

    def fit(self, X, y=None):
        if not hasattr(self, "column") or self.column is None:
            self.column = "elevation_m"
        from sklearn.preprocessing import PowerTransformer
        self.pt_ = PowerTransformer(method='yeo-johnson')
        self.pt_.fit(X[[self.column]])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.column] = self.pt_.transform(X[[self.column]])
        return X


class CityMedianImputer(BaseEstimator, TransformerMixin):
    """Impute numeric columns with city-level medians, falling back to global median."""
    def __init__(self, num_cols=None, city_col=None):
        self.num_cols = num_cols
        self.city_col = city_col

    def fit(self, X, y=None):
        if not hasattr(self, "num_cols") or self.num_cols is None:
            self.num_cols = []
        if not hasattr(self, "city_col") or self.city_col is None:
            self.city_col = "city"
        df = X[[self.city_col] + self.num_cols].copy()
        self.city_medians_ = (
            df.groupby(self.city_col)[self.num_cols]
              .median(numeric_only=True)
        )
        self.global_medians_ = df[self.num_cols].median(numeric_only=True)
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.num_cols:
            city_values = X[self.city_col].map(self.city_medians_[col])
            X[col] = X[col].fillna(city_values)
            X[col] = X[col].fillna(self.global_medians_[col])
        return X


class CityModeImputer(BaseEstimator, TransformerMixin):
    """Impute a categorical column with city-level mode, then global mode."""
    def __init__(self, col=None, city_col=None):
        self.col = col
        self.city_col = city_col

    def fit(self, X, y=None):
        if not hasattr(self, "col") or self.col is None:
            self.col = "soil_group"  # or 'storm_drain_type'
        if not hasattr(self, "city_col") or self.city_col is None:
            self.city_col = "city"
        df = X[[self.city_col, self.col]].copy()
        self.city_mode_ = (
            df.dropna(subset=[self.col])
              .groupby(self.city_col)[self.col]
              .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        )
        self.global_mode_ = df[self.col].mode().iloc[0] if not df[self.col].mode().empty else np.nan
        return self

    def transform(self, X):
        X = X.copy()
        mapped = X[self.city_col].map(self.city_mode_)
        X[self.col] = X[self.col].fillna(mapped)
        X[self.col] = X[self.col].fillna(self.global_mode_)
        return X


class ElevationKNNImputer(BaseEstimator, TransformerMixin):
    """KNN imputation for elevation using (lat, lon, elevation)."""
    def __init__(self, lat_col=None, lon_col=None, elev_col=None, n_neighbors=5):
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.elev_col = elev_col
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        if not hasattr(self, "lat_col") or self.lat_col is None:
            self.lat_col = "latitude"
        if not hasattr(self, "lon_col") or self.lon_col is None:
            self.lon_col = "longitude"
        if not hasattr(self, "elev_col") or self.elev_col is None:
            self.elev_col = "elevation_m"
        if not hasattr(self, "n_neighbors"):
            self.n_neighbors = 5
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors)
        self.cols_ = [self.lat_col, self.lon_col, self.elev_col]
        self.imputer_.fit(X[self.cols_])
        return self

    def transform(self, X):
        X = X.copy()
        imputed = self.imputer_.transform(X[self.cols_])
        X[self.elev_col] = imputed[:, 2]  # third column is elevation
        return X


class ElevationLocalDelta(BaseEstimator, TransformerMixin):
    """
    elev_local_delta_knn = elevation - mean(elevation of k nearest neighbours
    in the same catchment, using haversine distance on (lat, lon).
    For transform-time data, neighbours are taken from the training data.
    """
    def __init__(self, catchment_col=None, lat_col=None, lon_col=None, elev_col=None, k=8):
        self.catchment_col = catchment_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.elev_col = elev_col
        self.k = k

    def fit(self, X, y=None):
        if not hasattr(self, "catchment_col") or self.catchment_col is None:
            self.catchment_col = "catchment_id"
        if not hasattr(self, "lat_col") or self.lat_col is None:
            self.lat_col = "latitude"
        if not hasattr(self, "lon_col") or self.lon_col is None:
            self.lon_col = "longitude"
        if not hasattr(self, "elev_col") or self.elev_col is None:
            self.elev_col = "elevation_m"
        if not hasattr(self, "k"):
            self.k = 8
        # store training geo info
        self.train_geo_ = X[[self.catchment_col, self.lat_col, self.lon_col, self.elev_col]].copy()
        return self

    def transform(self, X):
        X = X.copy()
        X['elev_local_delta_knn'] = np.nan

        train_geo = self.train_geo_
        k = self.k

        for cid, g_te in X.groupby(self.catchment_col):
            g_tr = train_geo[train_geo[self.catchment_col] == cid]

            # Case A: catchment appears in training
            if len(g_tr) >= 1:  # changed from >=2 since k can be 1
                ref_coords = np.deg2rad(g_tr[[self.lat_col, self.lon_col]].to_numpy())
                qry_coords = np.deg2rad(g_te[[self.lat_col, self.lon_col]].to_numpy())

                k_eff = min(k, len(g_tr))
                if k_eff <= 0:
                    continue

                nn = NearestNeighbors(
                    n_neighbors=k_eff,
                    algorithm='ball_tree',
                    metric='haversine'
                ).fit(ref_coords)

                idxs = nn.kneighbors(qry_coords, return_distance=False)
                ref_elev = g_tr[self.elev_col].to_numpy()
                neigh_mean = np.array([ref_elev[i].mean() for i in idxs])

                test_elev = g_te[self.elev_col].to_numpy()
                deltas = test_elev - neigh_mean

                X.loc[g_te.index, 'elev_local_delta_knn'] = deltas

            # Case B: catchment only in transform data → within-transform k-NN
            else:
                n = len(g_te)
                if n <= 1:
                    continue

                coords = np.deg2rad(g_te[[self.lat_col, self.lon_col]].to_numpy())
                k_eff = min(k, n - 1)
                nn = NearestNeighbors(
                    n_neighbors=k_eff + 1,
                    algorithm='ball_tree',
                    metric='haversine'
                ).fit(coords)
                idxs = nn.kneighbors(coords, return_distance=False)[:, 1:]
                elev = g_te[self.elev_col].to_numpy()
                neigh_mean = np.array([elev[i].mean() for i in idxs])
                X.loc[g_te.index, 'elev_local_delta_knn'] = elev - neigh_mean

        return X


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drop specified columns (if present)."""
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        if not hasattr(self, "columns") or self.columns is None:
            self.columns = []
        return self

    def transform(self, X):
        return X.drop(columns=[c for c in self.columns if c in X.columns], errors="ignore")
