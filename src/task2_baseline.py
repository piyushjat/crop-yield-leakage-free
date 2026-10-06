"""Task 2: reproduce the paper's SEM baseline (random 80:20 split).

Runs on both merged files:
  paper-style : 28,242 rows with duplicated (area, item, year) keys
  clean       : 13,130 unique rows (temperature averaged per country-year)
Reports individual models, the paper-style stack (in-sample meta-features) and
a standard out-of-fold stack, plus how many test rows have a twin in train.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from yield_lib import (SEED, TARGET, FeatureBuilder, OOFStack, PaperStyleStack,
                       base_learners, metrics)

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/outputs")
FAST = "--fast" in sys.argv


def run(name, df):
    out = {}
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
    key = ["area", "item", "year"]
    twins = te.merge(tr[key].drop_duplicates(), on=key, how="inner")
    series_in_train = te.merge(tr[["area", "item"]].drop_duplicates(), on=["area", "item"])
    out["n_train"], out["n_test"] = len(tr), len(te)
    out["test_rows_with_same_area_item_year_in_train_pct"] = round(100 * len(twins) / len(te), 1)
    out["test_rows_with_same_area_item_series_in_train_pct"] = round(100 * len(series_in_train) / len(te), 1)

    fb = FeatureBuilder().fit(tr)
    Xtr, Xte = fb.transform(tr), fb.transform(te)
    ytr, yte = tr[TARGET].values, te[TARGET].values
    out["n_features"] = Xtr.shape[1]

    res = {}
    for mname, model in base_learners(FAST).items():
        t0 = time.time()
        res[mname] = metrics(yte, model.fit(Xtr, ytr).predict(Xte))
        print(f"[{name}] {mname}: R2={res[mname]['R2']:.3f} MAE={res[mname]['MAE']:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    res["SEM (paper-style stack)"] = metrics(yte, PaperStyleStack(FAST).fit(Xtr, ytr).predict(Xte))
    r = res["SEM (paper-style stack)"]
    print(f"[{name}] SEM paper-style: R2={r['R2']:.3f} MAE={r['MAE']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    t0 = time.time()
    res["SEM (out-of-fold stack)"] = metrics(yte, OOFStack(5, FAST).fit(Xtr, ytr).predict(Xte))
    r = res["SEM (out-of-fold stack)"]
    print(f"[{name}] SEM OOF: R2={r['R2']:.3f} MAE={r['MAE']:.3f} ({time.time()-t0:.0f}s)", flush=True)

    out["results"] = res
    return out


if __name__ == "__main__":
    results = {}
    for name, f in [("paper_style", "merged_paper_style.csv"), ("clean", "merged_clean.csv")]:
        results[name] = run(name, pd.read_csv(DATA / f))
        json.dump(results, open(DATA / "task2_results.json", "w"), indent=2)
    print("done")
