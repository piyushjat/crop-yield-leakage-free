"""Task 4: detrended yield-anomaly target under the Task 3 protocols.

Pipeline per fold (everything fitted on train rows only):
  1. fit per-series linear trend on train years (TrendModel, with fallbacks)
  2. target = yield - trend (in-sample residual on train rows)
  3. fit models on the anomaly; prediction = trend + predicted anomaly, clipped at 0
Feature sets
  A: year + climate levels + area/item one-hot   (same features as Tasks 2-3)
  B: climate DEVIATIONS from each series' own mean + crop one-hot (climate-driven signal only)
Models: Ridge, RF, XGB, SEM (out-of-fold stack). Baselines: trend only (anomaly = 0)
and the Task 3 naive baseline. 'Skill vs trend' = 1 - SSE_model / SSE_trend-only.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from yield_lib import (TARGET, ClimateDevBuilder, FeatureBuilder, OOFStack,
                       TrendModel, base_learners, metrics)
from task3_leakage_free import naive

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/outputs")
OUT = DATA / "task4_results.json"


def anomaly_models(Xtr, atr, Xte):
    bl = base_learners()
    return {
        "Ridge": Ridge(alpha=1.0).fit(Xtr, atr).predict(Xte),
        "RF": bl["RF"].fit(Xtr, atr).predict(Xte),
        "XGB": bl["XGB"].fit(Xtr, atr).predict(Xte),
        "SEM (out-of-fold stack)": OOFStack(5).fit(Xtr, atr).predict(Xte),
    }


def fit_predict_fold(tr, te):
    tm = TrendModel().fit(tr)
    trend_tr, trend_te = tm.predict(tr), tm.predict(te)
    sources = dict(tm.last_sources_)
    atr = tr[TARGET].values - trend_tr
    preds = {"Trend only (no climate)": trend_te}
    for tag, builder in [("A", FeatureBuilder()), ("B", ClimateDevBuilder())]:
        builder.fit(tr)
        Xtr, Xte = builder.transform(tr), builder.transform(te)
        for m, a_hat in anomaly_models(Xtr, atr, Xte).items():
            preds[f"{tag}: {m}"] = np.clip(trend_te + a_hat, 0, None)
    return preds, sources


def evaluate(df, splits, naive_kind):
    all_true, all_trend, all_pred, folds = [], [], {}, []
    for label, tri, tei in splits:
        t0 = time.time()
        tr, te = df.iloc[tri], df.iloc[tei]
        preds, sources = fit_predict_fold(tr, te)
        preds["Naive baseline"] = naive(tr, te, naive_kind)
        y = te[TARGET].values
        all_true.append(y)
        fold = {"fold": label, "n_train": len(tr), "n_test": len(te), "trend_source": sources}
        for m, p in preds.items():
            all_pred.setdefault(m, []).append(p)
            fold[m] = metrics(y, p)
        folds.append(fold)
        print(f"  fold {label}: trend-only R2={fold['Trend only (no climate)']['R2']:.3f} "
              f"A:RF R2={fold['A: RF']['R2']:.3f} B:RF R2={fold['B: RF']['R2']:.3f} "
              f"A:SEM R2={fold['A: SEM (out-of-fold stack)']['R2']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    y = np.concatenate(all_true)
    pooled = {m: metrics(y, np.concatenate(p)) for m, p in all_pred.items()}
    sse_trend = float(np.sum((y - np.concatenate(all_pred["Trend only (no climate)"])) ** 2))
    for m, p in all_pred.items():
        sse = float(np.sum((y - np.concatenate(p)) ** 2))
        pooled[m]["skill_vs_trend"] = 1 - sse / sse_trend
    return {"pooled": pooled, "folds": folds}


if __name__ == "__main__":
    df = pd.read_csv(DATA / "merged_clean.csv").reset_index(drop=True)
    results = json.load(open(OUT)) if OUT.exists() and "--resume" in sys.argv else {}

    if "temporal" not in results:
        print("Protocol: temporal holdout (train 1990-2005, test 2006-2013)", flush=True)
        tri, tei = np.where(df.year <= 2005)[0], np.where(df.year >= 2006)[0]
        results["temporal"] = evaluate(df, [("2006-2013", tri, tei)], "series_mean")
        json.dump(results, open(OUT, "w"), indent=2)

    if "country_out" not in results:
        print("Protocol: country-grouped 5-fold", flush=True)
        splits = [(f"countries-fold{k+1}", tr, te)
                  for k, (tr, te) in enumerate(GroupKFold(5).split(df, groups=df["area"]))]
        results["country_out"] = evaluate(df, splits, "item_mean")
        json.dump(results, open(OUT, "w"), indent=2)

    if "crop_out" not in results:
        print("Protocol: leave-one-crop-out", flush=True)
        splits = [(c, np.where(df.item != c)[0], np.where(df.item == c)[0])
                  for c in sorted(df.item.unique())]
        results["crop_out"] = evaluate(df, splits, "area_mean")
        json.dump(results, open(OUT, "w"), indent=2)
    print("done")
