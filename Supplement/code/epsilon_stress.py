"""Recompute the four smoothing baselines, 64 expiry configurations and positive control."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from importlib.metadata import version
import json
import platform
from pathlib import Path

from stress_core import HERE, FrozenGame, expiry_inputs, full_regrets, settings, two_player_run

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



if __name__ == "__main__":
    original=json.loads((HERE/'inputs'/'five_baseline.json').read_bytes())['inputs']
    results=[evaluate(('five',{},dict(original,epsilon=e),[])) for e in (100.,1000.)]
    output=HERE.parent/'results'/'epsilon_stress.json'
    record={'results':results,'provenance':{'solver':'COPT','coptpy':version('coptpy'),'python':platform.python_version()}}
    output.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf8')
    print([(r['epsilon'],r['decision'],r['lower'],r['checked_upper']) for r in results])
