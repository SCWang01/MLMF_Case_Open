"""Solve the 31 coalition market-participation models for each perturbed five-DER market."""
import argparse
from concurrent.futures import ProcessPoolExecutor
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
sys.path.insert(0, str(ROOT / "physical"))
from market_participation import solve

GROUPS = ("wind", "pv", "reference_load", "tg_capacity", "ess_rating", "price_level", "deviation_charge")


def factors():
    rng = np.random.default_rng(20261117)
    return [{key: 1 + 0.25 * (2 * rng.random() - 1) for key in GROUPS} for _ in range(10)]


def model(task):
    index, mask = task
    scale = factors()[index]
    with np.load(ROOT / "physical/inputs/five.npz") as data:
        params = {key: data[key].copy() for key in data.files}
    params["renewable"][0] *= scale["wind"]
    params["renewable"][1] *= scale["pv"]
    for group, fields in {"reference_load": ("dr_da", "dr_rt"), "tg_capacity": ("Pgmax",),
                          "ess_rating": ("p_cmax", "p_dmax", "Esmax"), "price_level": ("Lmpda", "Lmprt"),
                          "deviation_charge": ("pane",)}.items():
        for field in fields:
            params[field] = params[field] * scale[group]
    members = tuple(d + 1 for d in range(5) if mask >> d & 1)
    for seconds in (55, 240, 600):
        report = solve(params, members, seconds, 1, 1e-8)
        if report["lower"] is not None and report["upper"] is not None:
            gap = report["upper"] - report["lower"]
            if gap <= max(1e-5, 1e-7 * abs(report["lower"])):
                return index, mask, report
    raise ArithmeticError(f"Insufficient market participation bound for sample {index}, mask {mask}: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    target = HERE / "market_samples"
    target.mkdir(exist_ok=True)
    for index in range(10):
        rows = target / f"coalitions_{index}.jsonl"
        seen = {json.loads(line)["mask"] for line in rows.read_text().splitlines()} if rows.exists() else set()
        with ProcessPoolExecutor(max_workers=args.workers) as pool, rows.open("a",encoding="utf8") as stream:
            for _, mask, result in pool.map(model, ((index, m) for m in range(1,32) if m not in seen)):
                stream.write(json.dumps({"mask": mask, **result}) + "\n")
                stream.flush()
        records = {r["mask"]:r["lower"] for r in map(json.loads,rows.read_text().splitlines())}
        if set(records) != set(range(1,32)):
            raise ValueError(f"Incomplete market {index}")
        (target/f"data_{index}.json").write_text(json.dumps({"factors":factors()[index],"values":{0:0.0,**records}},indent=2))
        print("Completed market",index,flush=True)


if __name__ == "__main__":
    main()
