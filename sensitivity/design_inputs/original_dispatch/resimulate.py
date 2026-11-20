"""RESIMULATE Generate further simulated data sets with the project's dispatch model.

The manuscript's five-unit data are simulated, so more data can be simulated with the same model.
Each new data set scales seven groups of inputs by independent factors drawn uniformly from
[1 - SPREAD, 1 + SPREAD]; the dispatch problem of every one of the 31 coalitions is then solved again
with the accompanying historical dispatch model. These are upstream data-generation records.

Usage (interpreter with gurobipy, numpy, scipy):  python resimulate.py [WORKERS]
    resimulated/data_<k>.json - factors, the 32 coalition values, and the solver record of data set k

Invariant: a data set is written only if every coalition problem was solved to proven optimality
within the model's tolerances; otherwise the script stops and names the coalition.
Maintenance: edit only the "Define data" block to change the number of data sets, the spread, the
seed, or the scaled input groups. Existing files are kept, so an interrupted run resumes.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# ---------- Define data ----------------------------------------------------------------
HERE      = Path(__file__).resolve().parent
MODEL_DIR = HERE
OUT_DIR   = HERE / "resimulated"
NAMES     = ("WT", "PV", "TG", "ESS", "DR")
DATA_SETS = 10
SPREAD    = 0.25
SEED      = 20261117
# Input groups scaled together; the names are fields of the archived parameter structure.
GROUPS    = {"wind":             ("Wind_base", "Wind_f"),
             "pv":               ("Solar_base", "Solar_f"),
             "reference_load":   ("Load_base", "Load_f"),
             "tg_capacity":      ("Pgmax",),
             "ess_rating":       ("p_cmax", "p_dmax", "Esmax"),
             "price_level":      ("Lmpda", "Lmprt"),
             "deviation_charge": ("pane",)}
SOLVER    = {"seconds": 300.0, "threads": 2}
ACCEPTED  = "validated_numeric_optimum"

# ---------- Validate -------------------------------------------------------------------
if not (MODEL_DIR / "physical_model.py").exists():
    raise FileNotFoundError(f"resimulate:MissingModel {MODEL_DIR / 'physical_model.py'}")
if not 0 < SPREAD < 1:
    raise ValueError(f"resimulate:Spread {SPREAD} must lie strictly between 0 and 1")


# ---------- Derive ---------------------------------------------------------------------
def factors() -> list[dict]:
    """Factor sets of all data sets; the draw depends only on SEED, DATA_SETS, SPREAD and GROUPS."""
    rng = np.random.default_rng(SEED)
    return [{name: float(1 + SPREAD * (2 * rng.random() - 1)) for name in GROUPS} for _ in range(DATA_SETS)]


def solve(task: tuple[int, dict, int]) -> dict:
    """Solve the dispatch problem of one coalition under one factor set."""
    index, scale, mask = task
    import physical_model as pm
    base = pm.load_parameters()

    def scaled() -> dict:
        p = {name: value.copy() for name, value in base.items()}
        for group, fields in GROUPS.items():
            for field in fields:
                p[field] = p[field] * scale[group]
        return p

    original, pm.load_parameters = pm.load_parameters, scaled
    try:
        assets = [{"type": NAMES[d]} for d in range(len(NAMES)) if mask >> d & 1]
        with tempfile.TemporaryDirectory() as scratch:
            result = pm.solve_coalition(assets, Path(scratch), **SOLVER)
    finally:
        pm.load_parameters = original
    return {"data": index, "mask": mask, "status": result["status"], "value": result.get("lower_bound"),
            "bound": result.get("upper_bound"), "gap": result.get("absolute_gap"),
            "violation": result.get("physical_audit", {}).get("maximum_violation"),
            "seconds": result.get("elapsed_seconds")}


# ---------- Assemble output ------------------------------------------------------------
def main(workers: int) -> None:
    from multiprocessing import Pool
    OUT_DIR.mkdir(exist_ok=True)
    sets  = factors()
    tasks = [(k, sets[k], mask) for k in range(DATA_SETS) if not (OUT_DIR / f"data_{k}.json").exists()
             for mask in range(1, 1 << len(NAMES))]
    print(f"{len(tasks)} coalition problems to solve", flush=True)
    start, rows = time.monotonic(), {}
    with Pool(workers) as pool:
        for count, row in enumerate(pool.imap_unordered(solve, tasks), 1):
            if row["status"] != ACCEPTED:
                raise RuntimeError(f"main:NotOptimal data set {row['data']}, coalition mask {row['mask']}: {row['status']}")
            rows.setdefault(row["data"], {})[row["mask"]] = row
            if len(rows[row["data"]]) == (1 << len(NAMES)) - 1:
                k, done = row["data"], rows[row["data"]]
                table   = {"0": 0.0, **{str(mask): done[mask]["value"] for mask in sorted(done)}}
                (OUT_DIR / f"data_{k}.json").write_text(json.dumps({
                    "data": k, "factors": sets[k], "spread": SPREAD, "seed": SEED, "names": NAMES,
                    "values": table, "largest_gap": max(r["gap"] for r in done.values()),
                    "largest_violation": max(r["violation"] for r in done.values()),
                    "solver_seconds": sum(r["seconds"] for r in done.values())}, indent=1), encoding="utf-8")
                print(f"data set {k} written ({count}/{len(tasks)} problems, {time.monotonic() - start:.0f} s)", flush=True)
    print(f"finished in {time.monotonic() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
