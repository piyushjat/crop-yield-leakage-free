"""Robustness checks on the temporal protocol (train<=2005, test>=2006), clean file.

Part A  seeds      : raw-yield RF / XGB / SEM (out-of-fold) over 5 seeds (mean, sd)
Part B  tuning     : small time-aware search for RF and XGB (inner split: fit<=2000,
                     validate 2001-2005; test never touched), then refit on all train
                     years and evaluate on test; raw-yield and anomaly targets
Part C  threshold  : anomaly pipeline sensitivity to the trend min-points threshold
                     (5, 8, 12) for the full and no-climate feature sets
Everything is fitted on train years only.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from task4b_ablation import VARIANTS, features, sse
from yield_lib import TARGET, FeatureBuilder, TrendModel, metrics

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/outputs")
OUT = DATA / "task5_robustness_results.json"
CUT = 2005


def sem_oof(Xtr, ytr, Xte, seed):
    def bases():
        return [XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
                             random_state=seed, tree_method="hist"),
                RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=seed),
                DecisionTreeRegressor(random_state=seed),
                KNeighborsRegressor(n_neighbors=5, n_jobs=-1)]
    Z = np.zeros((len(ytr), 4))
    for a, b in KFold(5, shuffle=True, random_state=seed).split(Xtr):
        for j, m in enumerate(bases()):
            Z[b, j] = m.fit(Xtr[a], ytr[a]).predict(Xtr[b])
    fitted = [m.fit(Xtr, ytr) for m in bases()]
    meta = ExtraTreesRegressor(n_estimators=100, n_jobs=-1, random_state=seed).fit(Z, ytr)
    return meta.predict(np.column_stack([m.predict(Xte) for m in fitted]))


def summarize(vals):
    return {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}


if __name__ == "__main__":
    df = pd.read_csv(DATA / "merged_clean.csv").reset_index(drop=True)
    tr, te = df[df.year <= CUT], df[df.year > CUT]
    y, ytr = te[TARGET].values, tr[TARGET].values
    fb = FeatureBuilder().fit(tr)
    Xtr, Xte = fb.transform(tr), fb.transform(te)
    res = {}

    # ---------------------------------------------------------------- Part A
    print("Part A: seeds (raw yield, temporal cut 2005)", flush=True)
    A = {"RF": [], "XGB(subsample=0.8)": [], "SEM (out-of-fold stack)": []}
    for sd in [42, 43, 44, 45, 46]:
        t0 = time.time()
        rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=sd).fit(Xtr, ytr)
        A["RF"].append(metrics(y, rf.predict(Xte)))
        xg = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8,
                          n_jobs=-1, random_state=sd, tree_method="hist").fit(Xtr, ytr)
        A["XGB(subsample=0.8)"].append(metrics(y, xg.predict(Xte)))
        A["SEM (out-of-fold stack)"].append(metrics(y, sem_oof(Xtr, ytr, Xte, sd)))
        print(f"  seed {sd}: RF R2={A['RF'][-1]['R2']:.3f} SEM R2={A['SEM (out-of-fold stack)'][-1]['R2']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    res["A_seeds"] = {m: {k: summarize([r[k] for r in runs]) for k in ["R2", "MAE", "RMSE"]}
                      for m, runs in A.items()}
    json.dump(res, open(OUT, "w"), indent=2)

    # ---------------------------------------------------------------- Part B
    print("Part B: time-aware tuning", flush=True)
    inner_tr, inner_va = tr[tr.year <= 2000], tr[tr.year > 2000]
    fb_i = FeatureBuilder().fit(inner_tr)
    Xi_tr, Xi_va = fb_i.transform(inner_tr), fb_i.transform(inner_va)
    yi_tr, yi_va = inner_tr[TARGET].values, inner_va[TARGET].values

    def best_of(make, grid):
        best = None
        for combo in itertools.product(*grid.values()):
            params = dict(zip(grid, combo))
            p = make(params).fit(Xi_tr, yi_tr).predict(Xi_va)
            mae = float(np.mean(np.abs(yi_va - p)))
            if best is None or mae < best[0]:
                best = (mae, params)
        return best[1], best[0]

    rf_grid = {"max_depth": [None, 20], "min_samples_leaf": [1, 3, 5], "max_features": [1.0, 0.5]}
    xgb_grid = {"max_depth": [4, 6, 8], "learning_rate": [0.05, 0.1], "n_estimators": [200, 400]}
    rf_params, rf_val = best_of(lambda p: RandomForestRegressor(n_estimators=100, n_jobs=-1,
                                                                random_state=42, **p), rf_grid)
    xgb_params, xgb_val = best_of(lambda p: XGBRegressor(n_jobs=-1, random_state=42,
                                                         tree_method="hist", **p), xgb_grid)
    print(f"  best RF {rf_params} (val MAE {rf_val:.3f}); best XGB {xgb_params} (val MAE {xgb_val:.3f})", flush=True)

    mk_rf = lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42, **rf_params)
    mk_xgb = lambda: XGBRegressor(n_jobs=-1, random_state=42, tree_method="hist", **xgb_params)
    tm = TrendModel().fit(tr)
    trend_tr, trend_te = tm.predict(tr), tm.predict(te)
    atr = ytr - trend_tr
    B = {"best_params": {"RF": rf_params, "XGB": xgb_params}}
    for name, mk in [("RF tuned", mk_rf), ("XGB tuned", mk_xgb)]:
        raw = mk().fit(Xtr, ytr).predict(Xte)
        B[f"{name}: raw yield"] = metrics(y, raw)
        for vname in ["full (year+climate+area+item)", "no climate"]:
            cfg = VARIANTS[vname]
            Xa, Xb = features(tr, te, cfg["num"], cfg["cat"])
            p = np.clip(trend_te + mk().fit(Xa, atr).predict(Xb), 0, None)
            B[f"{name}: anomaly, {vname}"] = {**metrics(y, p),
                                             "skill_vs_trend": 1 - sse(y, p) / sse(y, trend_te)}
        print(f"  {name}: raw R2={B[f'{name}: raw yield']['R2']:.3f} MAE={B[f'{name}: raw yield']['MAE']:.3f}", flush=True)
    res["B_tuning"] = B
    json.dump(res, open(OUT, "w"), indent=2)

    # ---------------------------------------------------------------- Part C
    print("Part C: trend min-points threshold", flush=True)
    C = {}
    for mp in [5, 8, 12]:
        tmp = TrendModel(min_points=mp).fit(tr)
        t_tr, t_te = tmp.predict(tr), tmp.predict(te)
        has = np.array([(a, i) in tmp.series_ for a, i in zip(te.area, te.item)])
        a_tr = ytr - t_tr
        entry = {"fallback_test_rows": int((~has).sum()),
                 "trend_only": metrics(y, t_te)}
        for vname in ["full (year+climate+area+item)", "no climate", "climate only"]:
            cfg = VARIANTS[vname]
            Xa, Xb = features(tr, te, cfg["num"], cfg["cat"])
            rfm = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42).fit(Xa, a_tr)
            p = np.clip(t_te + rfm.predict(Xb), 0, None)
            entry[vname] = {**metrics(y, p), "skill_vs_trend": 1 - sse(y, p) / sse(y, t_te),
                            "skill_own_series_rows": 1 - sse(y[has], p[has]) / sse(y[has], t_te[has])}
        C[str(mp)] = entry
        print(f"  min_points={mp}: fallback rows={entry['fallback_test_rows']} trend-only R2={entry['trend_only']['R2']:.3f} | "
              f"skill full={entry['full (year+climate+area+item)']['skill_vs_trend']:+.3f} "
              f"no-climate={entry['no climate']['skill_vs_trend']:+.3f} "
              f"climate-only={entry['climate only']['skill_vs_trend']:+.3f} | own-series skill full="
              f"{entry['full (year+climate+area+item)']['skill_own_series_rows']:+.3f} "
              f"no-climate={entry['no climate']['skill_own_series_rows']:+.3f}", flush=True)
    res["C_threshold"] = C
    json.dump(res, open(OUT, "w"), indent=2)
    print("done")
