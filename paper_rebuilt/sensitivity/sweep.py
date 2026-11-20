"""Rebuild and solve the five-DER sensitivity families from COPT coalition profits."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

import os
CODE = Path(__file__).resolve().parent
HERE = Path(os.environ.get("MLMF_SENSITIVITY_OUTPUT", CODE)).resolve()
HERE.mkdir(exist_ok=True, parents=True)
ROOT = CODE.parents[1]
sys.path.insert(0, str(ROOT / "Supplement/code"))
sys.path.insert(0, str(ROOT / "large/core"))
sys.path.insert(0, str(ROOT / "small"))
from market_family import coefficients
from case20_game import MultiGame
from solve_minimax_4vpp import solve_minimax
from exact_audit import FrozenGame, full_regrets

RULES = ("equal_dp", "shapley")
LENGTHS = (5, 4)
GAMMA = (200.0, 250.0)


def market_specs(sample):
    if sample == "base":
        with (ROOT / "physical/five_tight.csv").open(newline="") as stream:
            values = {sum(1 << (int(x)-1) for x in r["members"].split(",")):float(r["lower"])
                      for r in csv.DictReader(stream)}
        values[0] = 0.
    else:
        data = json.loads((HERE / "market_samples" / f"data_{sample}.json").read_bytes())
        values = {int(k):v for k,v in data["values"].items()}
    if set(values) != set(range(32)):
        raise ValueError(f"Incomplete coalition values for {sample}")
    for first in range(1,31,2):
        if first.bit_count() not in (2,3):
            continue
        masks = (first,31 ^ first)
        for rules in product(RULES, repeat=2):
            f, s, base = coefficients(masks,rules,values)
            if sample != "base" and f.min() < 1:
                continue
            market = dict(f=f.tolist(), contribution=s.T.tolist(), base=base.tolist(), epsilon=0.1,
                          incumbent=[int(not (first >> d & 1)) for d in range(5)])
            yield masks,rules,market


def games(sample):
    for masks,rules,market in market_specs(sample):
        common = dict(masks=list(masks), rules=list(rules))
        for free in range(32):
            if sample == "base":
                yield dict(family="expiry", **common, free=free), market
            else:
                yield dict(family="resim",data=int(sample), **common,free=free), market
        if sample == "base" and masks == (13,18) and rules == RULES:
            for ages in product(*(range(LENGTHS[owner]) for owner in market["incumbent"])):
                yield dict(family="ages", **common,ages=list(ages)),market


def inputs_for(key, market):
    owners=market["incumbent"]
    deductions=np.zeros((5,2)); fees=np.zeros((2,5))
    for d,owner in enumerate(owners):
        rival=1-owner
        if key["family"] == "ages":
            periods = key["ages"][d]
            deductions[d,rival]=GAMMA[owner]*periods/LENGTHS[rival]
            fees[owner,d]=GAMMA[owner] if periods else 0
        elif not key["free"] >> d & 1:
            deductions[d,rival]=market["f"][d][rival]+1
    return dict(market,c=deductions.tolist(),compensation=fees.tolist())


def classify(key, market, repair=False):
    inputs=inputs_for(key,market)
    game=MultiGame(**{k: np.asarray(v) if k!="epsilon" else v for k,v in inputs.items()})
    exact=FrozenGame(inputs)
    starts = [(0,[0.5,0.5]),(0,[0.,1.]),(0,[1.,0.])] if repair else [(None,None),(0,[0.5,0.5])]
    for presolve, init in starts:
        result=solve_minimax(game, policy_mode="capture", tau=.001, decision_margin=1e-6,
                             seconds=180, round_seconds=30, absolute_gap=.001,max_rounds=200,
                             threads=1, progress=None, presolve=presolve,initial=init)
        low=result["lower_bound"]
        best=result["best_candidate"]
        quote=best["x"] if best else None
        upper=float(max(full_regrets(exact,tuple(map(Fraction,quote))))) if quote is not None else None
        decision = "unresolved"
        if upper is not None and upper<=.000999:
            decision="within"
        elif low>.001001 and (upper is None or low <= upper):
            decision="excluded"
        if (not repair or decision=="within") and decision!="unresolved" and (upper is None or low<=upper):
            break
    return dict(key=key,decision=decision,lower=low,upper=upper,quote=quote,
                rounds=len(result["history"]),presolve=presolve)


def batch(items):
    return [classify(*item) for item in items]


def canonical(key):
    return json.dumps(key,sort_keys=True,separators=(",",":"))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--samples",nargs="+",default=["base",*map(str,range(10))])
    parser.add_argument("--repair",action="store_true",help="Rerun decisions for which the independent interval bounds did not verify exclusion")
    args=parser.parse_args()
    dest=HERE/("repairs" if args.repair else "records")
    dest.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sample in args.samples:
            file=dest/f"{sample}.jsonl"
            completed={canonical(json.loads(line)["key"]) for line in file.read_text().splitlines()} if file.exists() else set()
            selected=None
            if args.repair:
                audit=HERE/"interval_audits"/f"{sample}.jsonl"
                if not audit.exists():
                    continue
                selected={canonical(r["key"]) for r in map(json.loads,audit.read_text().splitlines()) if r["lower"]<=.001001}
            items=[(key,market,args.repair) for key,market in games(sample)
                   if canonical(key) not in completed and (selected is None or canonical(key) in selected)]
            jobs=[pool.submit(batch,items[i:i+6]) for i in range(0,len(items),6)]
            with file.open("a",encoding="utf8") as stream:
                for job in as_completed(jobs):
                    for r in job.result():
                        stream.write(json.dumps(r,separators=(",",":"))+"\n")
                    stream.flush()
            print("Completed sensitivity family",sample,flush=True)


if __name__=="__main__":
    main()
