"""Recover bounds valid on the full ratio domain at comparable refinement sweeps.

The parent run records four ordinary CCG rounds. The refinement workers run
independently, so their split timestamps do not define a common iteration. At
sweep k we use each root's first k completed bisections (or its terminal
cover if it stopped earlier). The global lower bound is the minimum over
these 23 covers. A candidate can improve the feasible upper bound only after
its original run has evaluated it. The refinement routine also takes explicit
feasible warm starts; these first enter the trace at refinement sweep zero.
"""

import hashlib
import json
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import prod
from pathlib import Path
import sys


def global_trace(root):
    root = Path(root).resolve()
    sys.path.insert(0, str(root / "alternative"))
    from refine import check_cover
    from run_twenty import ExactGame, exact_upper

    def load(name):
        return json.loads((root / name).read_bytes())

    source = root / "physical/twenty_end_to_end.mat"
    exact = ExactGame(source)
    ccg = load("physical/large/results.json")
    summary = load("alternative/tightened/market.json")
    spec = load("large/data/refinement_rules.json")["boxes"]
    boxes = load("large/data/twenty_boxes.json")
    seed = load("verification/candidate_market.json")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if ccg["input_sha256"] != digest or summary["sha256"] != digest:
        raise ValueError("The parent CCG trace and refinements must use the same market.")
    if boxes.keys() != spec.keys() or boxes.keys() != summary["regions"].keys():
        raise ValueError("The initial cover differs between the saved runs.")

    # Endpoints in the saved partition are dyadic, so coverage can be checked exactly.
    volume = Fraction(0)
    for box in boxes.values():
        if len(box) != 4 or not all(0 <= lo < hi <= 1 for lo, hi in box):
            raise ValueError("Invalid initial root box.")
        volume += prod(Fraction(hi) - Fraction(lo) for lo, hi in box)
    for a, b in combinations(boxes.values(), 2):
        if all(max(al, bl) < min(ah, bh) for (al, ah), (bl, bh) in zip(a, b)):
            raise ValueError("Root interiors overlap.")
    if volume != 1:
        raise ValueError("Root boxes do not cover the unit hypercube.")

    @lru_cache(maxsize=None)
    def verified_regret(quote):
        if len(quote) != 4 or any(not 0 <= x <= 1 for x in quote):
            raise ValueError("A candidate is not a feasible four-VPP profile.")
        return exact_upper(exact, quote)

    saved = {}
    local_upper = {}
    for name, box in boxes.items():
        record = load(f"alternative/tightened/market/{name}.json")
        check_cover(record)
        steps = record["steps"]
        if steps[0]["bounds"] != box or spec[name]["bounds"] != box or record["name"] != name:
            raise ValueError(f"{name} has inconsistent initial bounds.")
        if len(steps) % 2 != 1 or steps[0]["path"] != "":
            raise ValueError(f"{name} is missing a root or a child.")
        frontier = {"": steps[0]["lower"]}
        if steps[0]["lower_of_cover"] != steps[0]["lower"]:
            raise ValueError(f"{name} has an invalid root bound.")
        for left, right in zip(steps[1::2], steps[2::2]):
            parent = left["path"][:-1]
            if not left["path"].endswith("L") or right["path"] != parent + "R" or parent not in frontier:
                raise ValueError(f"Unpaired child solves or inactive parent in {name}.")
            del frontier[parent]
            frontier[left["path"]] = left["lower"]
            frontier[right["path"]] = right["lower"]
            if min(frontier.values()) != right["lower_of_cover"]:
                raise ValueError(f"Incorrect intermediate complete-cover bound in {name}.")
        saved[name] = record
        starts = [tuple(spec[name]["audit_quote"])]
        if all(lo <= x <= hi for x, (lo, hi) in zip(seed["quote"], box)):
            starts.append(tuple(seed["quote"]))
        local_upper[name] = min(map(verified_regret, starts))

    rows = []
    upper = float("inf")
    for point in ccg["full_trace"]:
        upper = min(upper, point["upper"])
        rows.append(dict(iteration=len(rows) + 1, stage="full_domain", round=point["round"],
                         lower=point["lower"], upper=upper))

    max_sweep = max((len(row["steps"]) - 1) // 2 for row in saved.values())
    lower = rows[-1]["lower"]
    for sweep in range(max_sweep + 1):
        cover = []
        for name, record in saved.items():
            steps = record["steps"]
            end = 2 * sweep
            if end < len(steps):
                start = max(0, end - 1)
                candidates = (tuple(s["candidate"]) for s in steps[start:end + 1]
                              if s["candidate"] is not None)
                for candidate in candidates:
                    local_upper[name] = min(local_upper[name], verified_regret(candidate))
            cover.append(steps[min(end, len(steps) - 1)]["lower_of_cover"])
        current = min(cover)
        upper = min(upper, min(local_upper.values()))
        if not lower <= current <= upper:
            raise ValueError(f"Inconsistent bounds after global sweep {sweep}.")
        lower = current
        rows.append(dict(iteration=len(rows) + 1, stage="refinement", round=sweep,
                         lower=lower, upper=upper))
    for name, record in saved.items():
        if record["upper"] != local_upper[name]:
            raise ValueError(f"Failed to reproduce the feasible incumbent in {name}.")
    if (lower, upper) != (summary["lower"], summary["upper"]):
        raise ValueError("Final global bounds disagree with the COPT summary.")
    return {"trace": rows}
