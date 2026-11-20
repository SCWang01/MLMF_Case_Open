"""Check all Figure 5 keys using exact response witnesses or global deviation bounds."""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import time

from intervals import coefficients, exclude
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'small'))
from exact_audit import FrozenGame, full_regrets

HERE = Path(__file__).resolve().parent


def inputs_for(market, key):
    owners = market["incumbent"]
    deductions = [[0., 0.] for _ in owners]
    compensation = [[0.] * len(owners) for _ in range(2)]
    for d, owner in enumerate(owners):
        other = 1-owner
        if key["family"] == "ages":
            periods = key["ages"][d]
            gamma = (200., 250.)[owner]
            deductions[d][other] = gamma * periods / (5, 4)[other]
            compensation[owner][d] = gamma if periods else 0.
        elif not key["free"] & (1 << d):
            deductions[d][other] = market["f"][d][other] + 1.
    return {**market, "c": deductions, "compensation": compensation}


def evaluate(row, reference, market, cap):
    key = row["key"]
    if key != reference["key"]:
        raise ValueError("Mismatched configuration key")
    inputs = inputs_for(market, key)
    # The archive supplies only proposed actions for this check. No solver
    # decision or old master bound participates in the classification.
    witnesses = row.get("witnesses", [])
    profile = witnesses[0] if witnesses else reference["quote"]
    if profile is not None:
        x = tuple(map(Fraction, profile))
        if not all(0 <= q <= 1 for q in x):
            raise ValueError("Witness outside the feasible domain")
        regret = max(full_regrets(FrozenGame(inputs), x))
        if regret <= Fraction(999, 1000000):
            return "within", int(regret == 0), float(regret), 0
    lower, boxes = exclude(coefficients(inputs), cap)
    return ("excluded" if lower > .001001 else "unresolved"), 0, lower, boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-boxes", type=int, default=50000)
    args = parser.parse_args()
    design = json.loads((HERE/"design.json").read_bytes())
    refs = json.loads((HERE/"reference.json").read_bytes())
    if len(refs) != len(design["configurations"]):
        raise ValueError("Mismatched reference population")
    counts, ages, discrepancies = Counter(), Counter(), []
    started = time.monotonic()
    with (HERE/"outcomes.csv").open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["design_row","family","decision","exact_zero","checked_bound","boxes"])
        for i,(case,ref) in enumerate(zip(design["configurations"],refs)):
            key = case["key"]
            family = key["family"]
            decision,exact,bound,boxes = evaluate(case, ref, design["markets"][case["market"]], args.max_boxes)
            writer.writerow([i,family,decision,exact,bound,boxes])
            if i % 1000 == 0:
                out.flush()
                print(i, dict(counts), flush=True)
            counts[family,decision] += 1
            if family == "ages":
                period = key["ages"][1]
                ages[period,"exact"] += exact
                ages[period,"unresolved"] += decision == "unresolved"
                ages[period,"total"] += 1
            expected = {"approximate_equilibrium":"within","numerical_exclusion":"excluded"}[ref["decision"]]
            if expected != decision:
                discrepancies.append(i)
    groups = [dict(family=f, total=sum(counts[f,k] for k in ("within","excluded","unresolved")),
                   **{k:counts[f,k] for k in ("within","excluded","unresolved")}) for f in ("expiry","ages","resim")]
    result = dict(groups=groups, pv2=[dict(period=p, **{k:ages[p,k] for k in ("total","exact","unresolved")}) for p in range(4)],
                  discrepancy_rows=discrepancies, elapsed_seconds=time.monotonic()-started,
                  method="Exact rational full best-response checks at archived witnesses or feasible profiles; independent floating-point interval exclusion on the whole quote square, with 1e-6 margin.")
    (HERE/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result["groups"], flush=True)


if __name__ == "__main__":
    main()
