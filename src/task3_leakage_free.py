"""Task 3: leakage-free evaluation on the clean (de-duplicated) file.

Protocols
  temporal      : train 1990-2005, test 2006-2013
  country_out   : 5-fold GroupKFold by area (whole countries held out)
  crop_out      : leave-one-crop-out (10 folds, one crop held out each time)
Models: XGB, RF, DT, KNN, Ridge, SEM (paper-style stack), SEM (out-of-fold stack)
plus simple naive baselines. All fitted on train rows only (feature scaling and
category sets included). Yield in t/ha. Single seed, fixed hyperparameters.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from yield_lib import (TARGET, FeatureBuilder, OOFStack, PaperStyleStack,
                       base_learners, metrics)

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/outputs")
OUT = DATA / "task3_results.json"


def fit_predict_all(tr, te):
    """Fit every model on tr, predict te. Returns {model: predictions}."""
    fb = FeatureBuilder().fit(tr)
    Xtr, Xte = fb.transform(tr), fb.transform(te)
    ytr = tr[TARGET].values
    preds = {}
    for name, m in base_learners().items():
        preds[name] = m.fit(Xtr, ytr).predict(Xte)
    preds["Ridge"] = Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
    preds["SEM (paper-style stack)"] = PaperStyleStack().fit(Xtr, ytr).predict(Xte)
    preds["SEM (out-of-fold stack)"] = OOFStack(5).fit(Xtr, ytr).predict(Xte)
    return preds


def naive(tr, te, kind):
    """Naive baselines that use no climate information."""
    gmean = tr[TARGET].mean()
    if kind == "series_mean":  # same area+item mean, else item mean
        sm = tr.groupby(["area", "item"])[TARGET].mean()
        im = tr.groupby("item")[TARGET].mean()
        idx = pd.MultiIndex.from_frame(te[["area", "item"]])
        p = sm.reindex(idx).values
        fallback = te["item"].map(im).fillna(gmean).values
        return np.where(np.isnan(p), fallback, p)
    if kind == "item_mean":  # for held-out countries
        return te["item"].map(tr.groupby("item")[TARGET].mean()).fillna(gmean).values
    if kind == "area_mean":  # for held-out crops: country's mean over other crops
        return te["area"].map(tr.groupby("area")[TARGET].mean()).fillna(gmean).values


def evaluate(df, splits, naive_kind):
    """splits: list of (label, train_idx, test_idx). Returns pooled + per-fold metrics."""
    all_true, all_pred, folds = [], {}, []
    for label, tri, tei in splits:
        t0 = time.time()
        tr, te = df.iloc[tri], df.iloc[tei]
        preds = fit_predict_all(tr, te)
        preds["Naive baseline"] = naive(tr, te, naive_kind)
        all_true.append(te[TARGET].values)
        fold = {"fold": label, "n_train": len(tr), "n_test": len(te)}
        for m, p in preds.items():
            all_pred.setdefault(m, []).append(p)
            fold[m] = metrics(te[TARGET].values, p)
        folds.append(fold)
        print(f"  fold {label}: n_test={len(te)} SEM-OOF R2={fold['SEM (out-of-fold stack)']['R2']:.3f} "
              f"RF R2={fold['RF']['R2']:.3f} naive R2={fold['Naive baseline']['R2']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    y = np.concatenate(all_true)
    pooled = {m: metrics(y, np.concatenate(p)) for m, p in all_pred.items()}
    return {"pooled": pooled, "folds": folds}


if __name__ == "__main__":
    df = pd.read_csv(DATA / "merged_clean.csv").reset_index(drop=True)
    results = json.load(open(OUT)) if OUT.exists() and "--resume" in sys.argv else {}

    if "temporal" not in results:
        print("Protocol: temporal holdout (train 1990-2005, test 2006-2013)", flush=True)
        tri = np.where(df.year <= 2005)[0]
        tei = np.where(df.year >= 2006)[0]
        results["temporal"] = evaluate(df, [("2006-2013", tri, tei)], "series_mean")
        json.dump(results, open(OUT, "w"), indent=2)

    if "country_out" not in results:
        print("Protocol: country-grouped 5-fold (whole countries held out)", flush=True)
        splits = [(f"countries-fold{k+1}", tr, te)
                  for k, (tr, te) in enumerate(GroupKFold(5).split(df, groups=df["area"]))]
        results["country_out"] = evaluate(df, splits, "item_mean")
        json.dump(results, open(OUT, "w"), indent=2)

    if "crop_out" not in results:
        print("Protocol: leave-one-crop-out (10 folds)", flush=True)
        splits = [(c, np.where(df.item != c)[0], np.where(df.item == c)[0])
                  for c in sorted(df.item.unique())]
        results["crop_out"] = evaluate(df, splits, "area_mean")
        json.dump(results, open(OUT, "w"), indent=2)
    print("done")
