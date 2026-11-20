"""Recompute the four smoothing baselines, 64 expiry configurations and positive control."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from importlib.metadata import version
import json
import platform
from pathlib import Path

from core import HERE, FrozenGame, expiry_inputs, full_regrets, settings, two_player_run

def confirmed_profile(game, quote):
    """Find one rational equilibrium near the CCG quote, then check both global best responses."""
    point = tuple(map(Fraction, quote))
    if max(full_regrets(game, point)) == 0:
        return point
    # Intersections of response boundaries and regional first-order conditions
    # provide exact candidates. A failed witness search leaves the numerical
    # candidate unconfirmed and does not establish exclusion.
    points, segments, _ = game.candidates()
    for left, right in segments:
        points.update((left, right, tuple((a + b) / 2 for a, b in zip(left, right))))
    ordered = sorted(points, key=lambda p: (max(abs(a - b) for a, b in zip(p, point)), p))
    return next((p for p in ordered if max(full_regrets(game, p)) == 0), None)


def evaluate(task):
    case, key, inputs, witnesses = task
    row = two_player_run((case, key, inputs, witnesses))
    if row["decision"] == "unresolved":
        retry = two_player_run((case, key, inputs, witnesses), presolve=0)
        retry["first_attempt"] = row
        row = retry
    origin = "COPT"
    game = FrozenGame(inputs)
    row.update(case=case, epsilon=inputs["epsilon"], origin=origin)
    row["affiliation_probabilities"] = None
    if row["quote"] is not None:
        point = tuple(map(Fraction, row["quote"]))
        regret = max(full_regrets(game, point))
        if float(regret) != row["checked_upper"]:
            raise ValueError(f"Independent upper bound differs for {case} {key}")
        leaving = game.responses(point)
        row["affiliation_probabilities"] = [
            [float(1 - y), float(y)] if owner == 0 else [float(y), float(1 - y)]
            for owner, y in zip(inputs["incumbent"], leaving)
        ]
    row["exact_witness"] = None
    if row["decision"] == "approximate_equilibrium":
        point = None
        if witnesses:
            point = tuple(map(Fraction, witnesses[0]))
        elif row["quote"] is not None:
            point = confirmed_profile(game, row["quote"])
        if point is not None:
            if not all(0 <= x <= 1 for x in point) or max(full_regrets(game, point)) != 0:
                raise ValueError(f"Invalid exact witness for {case} {key}")
            row["exact_witness"] = list(map(str, point))
    # The count refers only to the witness included in this sweep record.
    row["verified_exact_profiles"] = int(row["exact_witness"] is not None)
    return row



def tasks():
    data = HERE.parent / "small" / "inputs"
    base = json.loads((data / "baseline.json").read_bytes())
    witnesses = json.loads((data / "expiry_witnesses.json").read_bytes())
    for epsilon in (0.01, 0.1, 1.0, 10.0):
        yield "five", {}, {**base["inputs"], "epsilon": epsilon}, []
    for epsilon in (0.1, 10.0):
        for mask in range(32):
            inputs = expiry_inputs(base["protected_states"][0]["inputs"], mask)
            inputs["epsilon"] = epsilon
            yield "expiry", {"free": mask}, inputs, witnesses.get(str(mask), []) if epsilon == 0.1 else []
    yield "control", {}, json.loads((HERE / "control_inputs.json").read_bytes()), []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, tasks()))
    if len(rows) != 69:
        raise ValueError("Incomplete ancillary runs")
    results = {"solver": "COPT", "coptpy_version": version("coptpy"), "python": platform.python_version(),
               "settings": {case:settings(case) for case in ("control","five","expiry")}, "results": rows}
    for e in (0.1,10.0):
        group = [r for r in rows if r["case"] == "expiry" and r["epsilon"] == e]
        print(e, dict(Counter(r["decision"] for r in group)), sum(bool(r["exact_witness"]) for r in group))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")

if __name__ == "__main__":
    main()
