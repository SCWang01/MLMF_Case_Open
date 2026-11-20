"""Compute the 2,043 published twenty-DER coalition values with native COPT."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PHYSICAL = HERE.parent / "physical"
sys.path.insert(0, str(PHYSICAL))
from dispatch import solve

FIELDS = ("members", "published", "lower", "upper", "status", "seconds", "time_limit")
INPUT = PHYSICAL / "inputs" / "twenty.npz"


def job(members, reference, limit):
    started = time.perf_counter()
    with np.load(INPUT) as src:
        params = dict(src)
    record = solve(params, tuple(map(int, members.split(","))), limit, 1, 1e-7)
    return dict(members=members, published=reference, lower=record["lower"], upper=record["upper"],
                status=record["status"], seconds=time.perf_counter()-started, time_limit=limit)


def relative_gap(row):
    if row["lower"] in (None,"") or row["upper"] in (None,""):
        return float("inf")
    return (float(row["upper"])-float(row["lower"])) / max(abs(float(row["lower"])),1.)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--time-limit",type=float,default=35)
    p.add_argument("--resume",action="store_true")
    args = p.parse_args()
    if args.workers < 1 or args.time_limit <= 0:
        p.error("positive workers and time limit required")
    refs = json.loads((PHYSICAL/"inputs/reference.json").read_bytes())["twenty"]
    output = HERE/"coalitions.csv"
    spec = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (INPUT,PHYSICAL/"inputs/reference.json",PHYSICAL/"dispatch.py")}
    manifest = HERE/"inputs_sha256.json"
    if args.resume and output.exists() and json.loads(manifest.read_bytes()) != spec:
        raise ValueError("Resume inputs or dispatch model changed")
    manifest.write_text(json.dumps(spec,indent=2),encoding="utf-8")
    finished = {}
    if args.resume and output.exists():
        with output.open(newline="") as file:
            for row in csv.DictReader(file):
                finished[row["members"]] = row
    pending = [(k,v) for k,v in refs.items() if k not in finished or relative_gap(finished[k])>1.01e-7]
    # Start large mixed-integer jobs early, then keep all workers occupied.
    pending.sort(key=lambda item:len(item[0].split(",")), reverse=True)
    with output.open("a" if finished else "w", newline="",encoding="utf-8") as file, ProcessPoolExecutor(max_workers=args.workers) as pool:
        out = csv.DictWriter(file,fieldnames=FIELDS)
        if not finished:
            out.writeheader()
        futures = [pool.submit(job,k,v,args.time_limit) for k,v in pending]
        for f in as_completed(futures):
            row = f.result()
            out.writerow(row)
            file.flush()
            finished[row["members"]]=row
    if set(finished) != set(refs):
        raise ValueError("Incomplete coalition coverage")
    feasible=[r for r in finished.values() if r["lower"] not in ("",None)]
    result=dict(coalitions=len(finished),feasible=len(feasible),
                max_relative_gap=max(relative_gap(r) for r in finished.values()),
                gap_above_target=[r["members"] for r in finished.values() if relative_gap(r)>1.01e-7],
                max_absolute_change=max(abs(float(r["lower"])-float(r["published"])) for r in feasible))
    (HERE/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result,flush=True)


if __name__ == "__main__":
    main()

