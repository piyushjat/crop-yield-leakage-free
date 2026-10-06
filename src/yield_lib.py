"""Reusable pieces for the crop-yield project (Tasks 2-4 and later phases)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

NUM_COLS = ["year", "rain_mm", "pesticides_tonnes", "avg_temp"]
CAT_COLS = ["area", "item"]
TARGET = "yield_t_ha"
SEED = 42


# --------------------------------------------------------------------- features
class FeatureBuilder:
    """One-hot encode area/item and standard-scale numerics.

    Fit on the TRAIN rows only, then transform train and test, so no test
    information (category set, scaling statistics) leaks into training.
    """

    def fit(self, df):
        self.cats_ = {c: sorted(df[c].unique()) for c in CAT_COLS}
        self.scaler_ = StandardScaler().fit(df[NUM_COLS])
        return self

    def transform(self, df):
        parts = [self.scaler_.transform(df[NUM_COLS]).astype(np.float32)]
        for c in CAT_COLS:
            vals = df[c].values
            parts.append(np.column_stack([(vals == v) for v in self.cats_[c]]).astype(np.float32))
        return np.hstack(parts)


# ----------------------------------------------------------------------- models
def base_learners(fast=False):
    n = 60 if fast else 100
    return {
        "XGB": XGBRegressor(n_estimators=n, max_depth=6, learning_rate=0.1, n_jobs=-1,
                            random_state=SEED, tree_method="hist"),
        "RF": RandomForestRegressor(n_estimators=n, n_jobs=-1, random_state=SEED),
        "DT": DecisionTreeRegressor(random_state=SEED),
        "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }


def meta_learner():
    return ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)


class PaperStyleStack:
    """Algorithm 1 of the paper: the meta-learner is trained on the base
    models' IN-SAMPLE predictions for the training data."""

    def __init__(self, fast=False):
        self.fast = fast

    def fit(self, X, y):
        self.bases_ = base_learners(self.fast)
        for m in self.bases_.values():
            m.fit(X, y)
        Z = np.column_stack([m.predict(X) for m in self.bases_.values()])
        self.meta_ = meta_learner().fit(Z, y)
        return self

    def predict(self, X):
        Z = np.column_stack([m.predict(X) for m in self.bases_.values()])
        return self.meta_.predict(Z)


class OOFStack:
    """Standard stacking: the meta-learner is trained on OUT-OF-FOLD base
    predictions, so it never sees predictions the bases made on their own
    training rows."""

    def __init__(self, n_splits=5, fast=False, groups=None):
        self.n_splits, self.fast, self.groups = n_splits, fast, groups

    def fit(self, X, y):
        y = np.asarray(y)
        names = list(base_learners(self.fast))
        Z = np.zeros((len(y), len(names)))
        for tr, va in KFold(self.n_splits, shuffle=True, random_state=SEED).split(X):
            for j, (name, m) in enumerate(base_learners(self.fast).items()):
                Z[va, j] = m.fit(X[tr], y[tr]).predict(X[va])
        self.bases_ = base_learners(self.fast)
        for m in self.bases_.values():
            m.fit(X, y)
        self.meta_ = meta_learner().fit(Z, y)
        return self

    def predict(self, X):
        Z = np.column_stack([m.predict(X) for m in self.bases_.values()])
        return self.meta_.predict(Z)


# ---------------------------------------------------------------------- metrics
def metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": r2_score(y_true, y_pred),
    }


# ------------------------------------------------------------- trend / anomaly
class TrendModel:
    """Per-series linear trend, fitted on TRAIN rows only (Task 4).

    Fallback chain for a test row whose own series has too few train years:
    series (area, item) -> crop (item) pooled trend -> country (area) pooled
    trend -> global pooled trend. Predictions are clipped at 0 (yield >= 0).
    """

    def __init__(self, min_points=8):
        self.min_points = min_points

    @staticmethod
    def _fit(g):
        if g["year"].nunique() < 2:
            return (float(g[TARGET].mean()), 0.0, float(g["year"].mean()))
        yr0 = float(g["year"].mean())
        b, a = np.polyfit(g["year"] - yr0, g[TARGET], 1)
        return (float(a), float(b), yr0)  # value = a + b * (year - yr0)

    def fit(self, tr):
        self.series_ = {k: self._fit(g) for k, g in tr.groupby(["area", "item"])
                        if len(g) >= self.min_points}
        self.item_ = {k: self._fit(g) for k, g in tr.groupby("item")}
        self.area_ = {k: self._fit(g) for k, g in tr.groupby("area")}
        self.global_ = self._fit(tr)
        return self

    def predict(self, df):
        out = np.empty(len(df))
        src = []
        for i, (a, it, y) in enumerate(zip(df["area"], df["item"], df["year"])):
            if (a, it) in self.series_:
                p, s = self.series_[(a, it)], "series"
            elif it in self.item_:
                p, s = self.item_[it], "item"
            elif a in self.area_:
                p, s = self.area_[a], "area"
            else:
                p, s = self.global_, "global"
            out[i] = p[0] + p[1] * (y - p[2])
            src.append(s)
        self.last_sources_ = pd.Series(src).value_counts().to_dict()
        return np.clip(out, 0, None)


CLIM_COLS = ["rain_mm", "pesticides_tonnes", "avg_temp"]


class ClimateDevBuilder:
    """Features = climate deviations from each series' own climate mean
    (+ crop one-hot). Series mean comes from train rows; for a series unseen in
    train it is the mean of that series' own (feature-only) rows in the frame
    being transformed. No target information is used."""

    def fit(self, tr):
        self.means_ = tr.groupby(["area", "item"])[CLIM_COLS].mean()
        self.items_ = sorted(tr["item"].unique())
        self.scaler_ = StandardScaler().fit(self._dev(tr))
        return self

    def _dev(self, df):
        own = df.groupby(["area", "item"])[CLIM_COLS].transform("mean")
        idx = pd.MultiIndex.from_frame(df[["area", "item"]])
        m = self.means_.reindex(idx)
        m.index = df.index
        m = m.fillna(own)
        return (df[CLIM_COLS] - m).values

    def transform(self, df):
        d = self.scaler_.transform(self._dev(df)).astype(np.float32)
        oh = np.column_stack([(df["item"].values == v) for v in self.items_]).astype(np.float32)
        return np.hstack([d, oh])
