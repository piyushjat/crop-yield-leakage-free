"""Task 1: load, clean and merge the four crop-yield datasets.

Produces two files:
  merged_paper_style.csv : replicates the paper's merge (28,242 rows). Contains
                           duplicated (area, item, year) keys because temp.csv
                           holds several station-level readings per country-year.
  merged_clean.csv       : temperature averaged per country-year BEFORE merging,
                           so each (area, item, year) appears exactly once.
Yield is converted from hg/ha to t/ha (divide by 10,000).
"""
import sys
from pathlib import Path

import pandas as pd

IN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/mnt/user-data/uploads")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/mnt/user-data/outputs")


def load():
    y = pd.read_csv(IN_DIR / "yield.csv")
    t = pd.read_csv(IN_DIR / "temp.csv")
    r = pd.read_csv(IN_DIR / "rainfall.csv")
    p = pd.read_csv(IN_DIR / "pesticides.csv")

    r.columns = r.columns.str.strip()  # ' Area' has a leading space
    r["average_rain_fall_mm_per_year"] = pd.to_numeric(
        r["average_rain_fall_mm_per_year"], errors="coerce"  # '..' and blanks -> NaN
    )
    for df, col in [(y, "Area"), (r, "Area"), (p, "Area"), (t, "country")]:
        df[col] = df[col].str.strip()

    y = y[["Area", "Item", "Year", "Value"]].rename(
        columns={"Area": "area", "Item": "item", "Year": "year", "Value": "yield_hg_ha"}
    )
    r = r.rename(columns={"Area": "area", "Year": "year",
                          "average_rain_fall_mm_per_year": "rain_mm"})
    p = p[["Area", "Year", "Value"]].rename(
        columns={"Area": "area", "Year": "year", "Value": "pesticides_tonnes"}
    )
    t = t.rename(columns={"country": "area"})
    return y, t, r, p


def merge(y, t, r, p):
    m = y.merge(p, on=["area", "year"]).merge(r, on=["area", "year"]).merge(t, on=["area", "year"])
    return m.dropna()


def main():
    y, t, r, p = load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paper = merge(y, t, r, p)
    paper["yield_t_ha"] = paper["yield_hg_ha"] / 10_000
    paper.to_csv(OUT_DIR / "merged_paper_style.csv", index=False)

    t_agg = t.groupby(["area", "year"], as_index=False)["avg_temp"].mean()
    clean = merge(y, t_agg, r, p)
    clean["yield_t_ha"] = clean["yield_hg_ha"] / 10_000
    clean = clean.sort_values(["area", "item", "year"]).reset_index(drop=True)
    assert not clean.duplicated(["area", "item", "year"]).any()
    clean.to_csv(OUT_DIR / "merged_clean.csv", index=False)

    for name, df in [("paper-style", paper), ("clean", clean)]:
        print(f"{name}: {len(df):,} rows | {df.area.nunique()} areas | {df.item.nunique()} items | "
              f"years {df.year.min()}-{df.year.max()} | "
              f"duplicate (area,item,year) rows: {df.duplicated(['area','item','year']).sum():,}")


if __name__ == "__main__":
    main()
