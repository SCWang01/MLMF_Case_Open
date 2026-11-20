"""Rerun the COPT CCG master on a reproducible sample, or on every design row."""
import argparse
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "large" / "core"))
from case20_game import MultiGame
from solve_minimax_4vpp import solve_minimax
from run import inputs_for, FrozenGame, full_regrets


def copt_check(task):
    index, inputs, expected, *settings = task
    game = MultiGame(**{k:np.asarray(v) if k != "epsilon" else v for k,v in inputs.items()})
    raw = solve_minimax(game, policy_mode="capture", tau=.001, decision_margin=1e-6,
                        seconds=300, round_seconds=30, absolute_gap=.001, max_rounds=200, threads=1, progress=None, **(settings[0] if settings else {}))
    lower, decision = raw["lower_bound"], raw["decision"]
    upper = None
    if raw["best_candidate"] is not None:
        x = tuple(map(Fraction, raw["best_candidate"]["x"]))
        upper = float(max(full_regrets(FrozenGame(inputs),x)))
        if lower > upper or (decision == "approximate_equilibrium" and upper > .000999):
            decision = "unresolved"
    return (index,expected,decision,lower,upper,len(raw["history"]),raw["elapsed_seconds"])


def select_rows(design, refs, per_stratum):
    # Stratify by family and paper decision, include every approximate-only
    # witness, and sample evenly across the entire published input order.
    strata = defaultdict(list)
    selected = set()
    for i,(row,ref) in enumerate(zip(design["configurations"],refs)):
        strata[row["key"]["family"],ref["decision"]].append(i)
        if ref["quote"] is not None:
            selected.add(i)
    for indices in strata.values():
        selected.update(indices[j] for j in np.linspace(0,len(indices)-1,
                        min(per_stratum,len(indices)),dtype=int))
    return sorted(selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-stratum", type=int, default=12)
    p.add_argument("--all", action="store_true")
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args()
    if min(args.workers,args.per_stratum)<1:
        p.error("workers and per-stratum must be positive")
    design=json.loads((HERE/"design.json").read_bytes())
    refs=json.loads((HERE/"reference.json").read_bytes())
    indices=range(len(refs)) if args.all else select_rows(design,refs,args.per_stratum)
    counts=Counter()
    tasks=[]
    for i in indices:
        row=design["configurations"][i]
        tasks.append((i,inputs_for(design["markets"][row["market"]],row["key"]),refs[i]["decision"]))
    filename="copt_full.csv" if args.all else "copt_crosscheck.csv"
    with ProcessPoolExecutor(max_workers=args.workers) as executor, (HERE/filename).open("w",newline="",encoding="utf-8") as file:
        writer=csv.writer(file)
        writer.writerow(["design_row","published","copt","lower","checked_upper","rounds","seconds"])
        for result in executor.map(copt_check,tasks):
            writer.writerow(result)
            file.flush()
            counts[result[1],result[2]]+=1
    print(dict(counts),flush=True)


if __name__=="__main__":
    multiprocessing.freeze_support()
    main()

