"""Evaluate two fixed rules against recorded outcomes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def score(frame, loss="absolute"):
    difference=frame[["baseline", "candidate"]].sub(frame["actual"], axis=0)
    errors=difference.abs() if loss=="absolute" else difference.pow(2)
    gap=errors["candidate"]-errors["baseline"]
    if frame.empty or not np.isfinite(errors.to_numpy()).all():
        raise ValueError("No finite scores for the requested benchmark")
    worst=gap.idxmax()
    return {
        "cases":len(frame),
        "baseline_loss":round(float(errors["baseline"].mean()), 6),
        "candidate_loss":round(float(errors["candidate"].mean()), 6),
        "candidate_wins":int((gap < -1e-9).sum()),
        "baseline_wins":int((gap > 1e-9).sum()),
        "ties":int((gap.abs() <= 1e-9).sum()),
        "largest_candidate_disadvantage":round(float(gap.loc[worst]), 6),
        "worst_case":str(worst),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    try:
        result=evaluate(args.input)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    except (OSError, ValueError, KeyError, ImportError) as error:
        parser.exit(2,f"Observed benchmark failed: {error}\n")


def evaluate(path):
    data=pd.read_csv(path,low_memory=False)
    rows=[]
    for year in range(2006,2024):
        prior,later=year-1,year+1
        subset=data.loc[
            data["LI_DATE"].eq(year)
            & ~data["CWI_DATE"].between(prior,later)
            & ~data["PV_DATE"].between(prior,later)
            & data[f"GasValFlag{prior}"].eq("V")
            & data[f"GasValFlag{later}"].eq("V"),
            ["PROP_TYPE",f"Gcons{prior}",f"Gcons{later}"],
        ].copy()
        subset.columns=["property_type","before","after"]
        subset=subset.dropna().query("before > 0 and after > 0")
        subset["year"]=year
        rows.append(subset)
    panel=pd.concat(rows,ignore_index=True)
    panel["fraction"]=(panel.before-panel.after)/panel.before
    train=panel.loc[panel.year.le(2015)]
    test=panel.loc[panel.year.ge(2018)].copy()
    pooled_fraction=train.fraction.median()
    fractions=train.groupby("property_type").fraction.agg(["size","median"])
    eligible=fractions.loc[fractions["size"].ge(30),"median"]
    test["baseline"]=test.before*pooled_fraction
    test["candidate"]=test.before*test.property_type.map(eligible).fillna(pooled_fraction)
    test["actual"]=test.before-test.after
    test=test.reset_index(drop=True)
    test.index=[f"{row.year} / {row.property_type} / case {i+1}" for i,row in test.iterrows()]
    return {"source":"DESNZ NEED 2026 anonymised 50,000-property sample","metric":"MAE in kWh/year",
            "baseline":"Pooled median proportional gas change","candidate":"Median proportional gas change by property type",
            "fit_through_install_year":2015,"training_properties":len(train),"holdout_install_years":[2018,2023],
            "metrics":score(test),"by_property_type":{str(k):score(v) for k,v in test.groupby("property_type")}}


if __name__=="__main__":
    main()
