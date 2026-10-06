"""Task 4 diagnostic: where does the anomaly model's skill over the trend come from?

Temporal protocol, two cuts (train<=2005 / test>=2006 and train<=2000 / test>=2001).
Per cut: fit per-series trend on train years, model the anomaly with different
feature groups, add the trend back, clip at 0. Skill vs trend = 1 - SSE_model / SSE_trend.
RF is repeated over 3 seeds (mean and sd); Ridge and XGB use one run.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from yield_lib import TARGET, TrendModel, metrics

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/outputs")
CLIM = ["rain_mm", "pesticides_tonnes", "avg_temp"]
VARIANTS = {
    "full (year+climate+area+item)": dict(num=["year"] + CLIM, cat=["area", "item"]),
    "no climate": dict(num=["year"], cat=["area", "item"]),
    "no year": dict(num=CLIM, cat=["area", "item"]),
    "no area/item (year+climate)": dict(num=["year"] + CLIM, cat=[]),
    "climate only": dict(num=CLIM, cat=[]),
    "year only": dict(num=["year"], cat=[]),
    "area+item only": dict(num=[], cat=["area", "item"]),
}
CUTS = {"train<=2005, test>=2006": 2005, "train<=2000, test>=2001": 2000}
SEEDS = [42, 43, 44]


def features(tr, te, num, cat):
    parts_tr, parts_te = [], []
    if num:
        sc = StandardScaler().fit(tr[num])
        parts_tr.append(sc.transform(tr[num]))
        parts_te.append(sc.transform(te[num]))
    for c in cat:
        vals = sorted(tr[c].unique())
        parts_tr.append(np.column_stack([tr[c].values == v for v in vals]).astype(float))
        parts_te.append(np.column_stack([te[c].values == v for v in vals]).astype(float))
    return np.hstack(parts_tr).astype(np.float32), np.hstack(parts_te).astype(np.float32)


def sse(y, p):
    return float(np.sum((y - p) ** 2))


if __name__ == "__main__":
    df = pd.read_csv(DATA / "merged_clean.csv").reset_index(drop=True)
    results = {}
    for cut_name, cut in CUTS.items():
        tr, te = df[df.year <= cut], df[df.year > cut]
        y = te[TARGET].values
        tm = TrendModel().fit(tr)
        trend_tr, trend_te = tm.predict(tr), tm.predict(te)
        atr = tr[TARGET].values - trend_tr
        sse_trend = sse(y, trend_te)
        res = {"n_train": len(tr), "n_test": len(te),
               "Trend only": {**metrics(y, trend_te), "skill_vs_trend": 0.0}}
        print(f"\n{cut_name}: n_train={len(tr)} n_test={len(te)} trend-only R2={res['Trend only']['R2']:.3f} "
              f"MAE={res['Trend only']['MAE']:.3f}", flush=True)
        for vname, cfg in VARIANTS.items():
            Xtr, Xte = features(tr, te, cfg["num"], cfg["cat"])
            out = {}
            runs = []
            for sd in SEEDS:
                rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=sd).fit(Xtr, atr)
                p = np.clip(trend_te + rf.predict(Xte), 0, None)
                runs.append((metrics(y, p), 1 - sse(y, p) / sse_trend))
            out["RF"] = {
                "R2_mean": float(np.mean([r[0]["R2"] for r in runs])),
                "MAE_mean": float(np.mean([r[0]["MAE"] for r in runs])),
                "skill_mean": float(np.mean([r[1] for r in runs])),
                "skill_sd": float(np.std([r[1] for r in runs])),
            }
            xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
                               random_state=42, tree_method="hist").fit(Xtr, atr)
            p = np.clip(trend_te + xgb.predict(Xte), 0, None)
            out["XGB"] = {**metrics(y, p), "skill_vs_trend": 1 - sse(y, p) / sse_trend}
            rg = Ridge(alpha=1.0).fit(Xtr, atr)
            p = np.clip(trend_te + rg.predict(Xte), 0, None)
            out["Ridge"] = {**metrics(y, p), "skill_vs_trend": 1 - sse(y, p) / sse_trend}
            res[vname] = out
            print(f"  {vname:32s} RF skill={out['RF']['skill_mean']:+.3f}±{out['RF']['skill_sd']:.3f} "
                  f"(R2={out['RF']['R2_mean']:.3f}, MAE={out['RF']['MAE_mean']:.3f}) | "
                  f"XGB skill={out['XGB']['skill_vs_trend']:+.3f} | Ridge skill={out['Ridge']['skill_vs_trend']:+.3f}",
                  flush=True)
        results[cut_name] = res
        json.dump(results, open(DATA / "task4b_ablation_results.json", "w"), indent=2)
    print("done")
