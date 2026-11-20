"""Extract the manuscript panel data from six frozen-input solve and audit records."""

import argparse
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
INPUTS = (
    "small/results.json", "large/results.json", "sensitivity/copt_results.json",
    "sensitivity/results.json", "alternative/refinement/frozen.json",
    "alternative/search/comparison.json",
)


def extract(case_root):
    records, hashes = [], {}
    for name in INPUTS:
        raw = (case_root / name).read_bytes()
        hashes[name] = sha256(raw).hexdigest()
        records.append(json.loads(raw))
    five, twenty, sensitivity, exact, refinement, comparison = records
    integer = comparison["frozen"]
    if not twenty["input_sha256"] == refinement["sha256"] == integer["source_sha256"]:
        raise ValueError("The regional and full-domain runs use different games")
    boxes = twenty["boxes"]
    if len({b["box"] for b in boxes}) != 23 or len(boxes) != 23:
        raise ValueError("Expected all 23 manuscript cover boxes")
    if set(refinement["regions"]) != {b["box"] for b in boxes}:
        raise ValueError("The refinement must cover the same boxes")
    if refinement["lower"] != min(r["lower"] for r in refinement["regions"].values()):
        raise ValueError("The global refinement bound differs from the weakest box")
    if integer["lower"] != float(Fraction(integer["exact_lower"])):
        raise ValueError("The integer bound differs from the exact fraction")
    decision_threshold = twenty["tau"] + twenty["decision_margin"]
    if not decision_threshold < refinement["lower"] <= integer["lower"] <= integer["upper"]:
        raise ValueError("Invalid global bounds or exclusion threshold")

    first_exclusions = []
    for box in boxes:
        if box["decision"] not in ("numerical_exclusion", "unresolved"):
            raise ValueError(f"Unexpected decision for {box['box']}")
        first = next((r["round"] for r in box["trace"] if r["lower"] > decision_threshold), None)
        if (first is None) != (box["decision"] == "unresolved"):
            raise ValueError(f"Decision and lower-bound trace disagree for {box['box']}")
        if first is not None:
            first_exclusions.append(first)
    last_round = max(b["trace"][-1]["round"] for b in boxes)
    by_round = [{"round": i, "excluded": sum(first <= i for first in first_exclusions)}
                for i in range(1, last_round + 1)]
    states = {state["expired"]: state["regrets"] for state in five["protected_states"]}
    settings = five["decision_settings"]
    first = next(r["round"] for r in five["trace"] if r["lower"] > settings["tau"] + settings["decision_margin"])
    if first != five["first_exclusion_round"]:
        raise ValueError("Five-DER decision and refinement runs first exclude at different rounds")
    return {
        "sources_sha256": hashes,
        "solver": f"COPT {twenty['solver']}",
        "case": "Frozen manuscript payoff coefficients",
        "five": {
            "gains": [{"expired": state, "vppo": states[state]} for state in ("None", "ESS", "All")],
            "regret_bounds": [{k: row[k] for k in ("round", "lower", "upper")} for row in five["trace"]],
            "first_exclusion_round": first,
        },
        "twenty": {
            "procedure": "Endpoint-initialized CCG",
            "full_domain": [{k: r[k] for k in ("round", "upper")} for r in twenty["full_trace"]],
            "cover": by_round,
            "total_boxes": len(boxes),
            "unresolved_boxes": len(boxes) - len(first_exclusions),
            "global_bounds": {
                "decision_threshold": decision_threshold,
                "copt_refinement": refinement["lower"],
                "integer_certificate": integer["exact_lower"],
            },
        },
        "sensitivity": {
            "groups": sensitivity["groups"], "pv2": exact["pv2"],
            "methods": {"groups": sensitivity["method"], "pv2": exact["method"]},
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "figure_data.json")
    args = parser.parse_args()
    args.output.write_text(json.dumps(extract(args.case_root), indent=2, allow_nan=False) + "\n", encoding="utf-8")
