"""Evaluate all expiry subsets under the protected five-DER fee schedule."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from stress_core import expiry_inputs, two_player_run

HERE = Path(__file__).resolve().parent
PROTECTION_FACTOR = 41.46247873622652


def solve_mask(task):
    mask, protected = task
    inputs = expiry_inputs(protected, mask)
    run = ("expiry", {"mask": mask}, inputs, [])
    row = two_player_run(run)
    if row["decision"] == "unresolved":
        row = two_player_run(run, presolve=0)
    if row["decision"] == "unresolved":
        raise RuntimeError(f"Unresolved expiry mask {mask}")
    return mask, row["decision"]


if __name__ == "__main__":
    baseline = json.loads((HERE / "inputs/five_baseline.json").read_bytes())["inputs"]
    protected = dict(baseline)
    for field in ("c", "compensation"):
        protected[field] = [[entry * PROTECTION_FACTOR for entry in row] for row in baseline[field]]
    with ProcessPoolExecutor(max_workers=3) as workers:
        decisions = dict(workers.map(solve_mask, ((m, protected) for m in range(32))))
    out = HERE.parent / "results/expiry_subsets.json"
    out.write_text(json.dumps(decisions, indent=2) + "\n", encoding="utf-8")
