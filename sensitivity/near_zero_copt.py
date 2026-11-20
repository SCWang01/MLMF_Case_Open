"""Reproduce and independently check the 15 near-zero sensitivity candidate quotes."""
import csv
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
from run import inputs_for, FrozenGame, full_regrets

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "large/core"))
from case20_game import MultiGame
from solve_minimax_4vpp import solve_minimax


def main():
    read = lambda p: json.loads(p.read_bytes())
    key = lambda value: json.dumps(value, sort_keys=True, separators=(",",":"))
    design = read(HERE / "design.json")
    index = {key(row["key"]): i for i,row in enumerate(design["configurations"])}
    cases = read(HERE.parent / "ancillary/near_zero_cases.json")
    with (HERE / "copt_audited.csv").open(newline="") as f:
        rows = {int(r["design_row"]):r for r in csv.DictReader(f)}
    checked = []
    for case in cases:
        i = index[key(case["key"])]
        row = rows[i]
        item = design["configurations"][i]
        inputs = inputs_for(design["markets"][item["market"]], item["key"])
        if inputs != case["inputs"]:
            raise ValueError(f"Frozen inputs differ for {case['id']}")
        game = MultiGame(**{k:np.asarray(v) if k != "epsilon" else v for k,v in inputs.items()})
        raw = solve_minimax(game, policy_mode="capture", tau=.001, decision_margin=1e-6,
                            seconds=300, round_seconds=30, absolute_gap=.001, max_rounds=200,
                            threads=1, progress=None, presolve=0 if row["presolve"]=="off" else None)
        if not raw["best_candidate"] or raw["decision"] != "approximate_equilibrium":
            raise ArithmeticError(f"No COPT approximate equilibrium at {case['id']}")
        quote = raw["best_candidate"]["x"]
        gains = full_regrets(FrozenGame(inputs), tuple(map(Fraction, quote)))
        bound = max(gains)
        if not 0 < bound <= Fraction(999, 1000000):
            raise ArithmeticError(f"Failed response audit for {case['id']}")
        checked.append(dict(id=case["id"], design_row=i, key=item["key"], presolve=row["presolve"],
                            quote=quote, gains=[str(r) for r in gains], checked_upper=float(bound),
                            exact_checked_upper=str(bound), full_sweep_upper=float(row["checked_upper"])))
    if len(checked)!=15 or len({r["design_row"] for r in checked})!=15:
        raise ValueError("Incomplete near-zero cases")
    (HERE / "near_zero_copt.json").write_text(json.dumps(checked,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("COPT full-sweep maximum", max(r["full_sweep_upper"] for r in checked),
          "rerun maximum", max(r["checked_upper"] for r in checked))

if __name__ == "__main__":
    main()

