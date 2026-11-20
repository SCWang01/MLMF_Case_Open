"""Run the four-VPP COPT experiment. Every optimization starts from case inputs."""
import argparse
import hashlib
import json
import math
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
import coptpy
import numpy
import scipy
from audit_minimax_4vpp_candidate import ExactGame
from solve_minimax_4vpp import read_game, solve_minimax

SOURCE = ROOT.parent / "physical/twenty_end_to_end.mat"
BOX_FILE = ROOT / "data" / "twenty_boxes.json"
TAU = 0.001
DELTA = 1e-6


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def validate_cover(boxes):
    bounds = [[(Fraction(a), Fraction(b)) for a, b in box] for box in boxes.values()]
    if not all(len(box) == 4 and all(0 <= a < b <= 1 for a, b in box) for box in bounds):
        raise ValueError("Invalid box bounds")
    if sum(math.prod(b - a for a, b in box) for box in bounds) != 1:
        raise ValueError("Cover volume differs from one")
    for first, second in combinations(bounds, 2):
        if all(max(a, c) < min(b, d) for (a, b), (c, d) in zip(first, second)):
            raise ValueError("Cover has overlapping interiors")


def check_candidate(quote, source=SOURCE):
    exact = ExactGame(source)
    point = tuple(map(Fraction, quote))
    shares = exact.shares(point)
    payoffs = [exact.payoff(i, point, shares) for i in range(4)]
    best = [exact.best_response(i, point) for i in range(4)]
    regrets = [float(best[i][1] - payoffs[i]) for i in range(4)]
    return dict(quote=quote, regrets=regrets, checked_upper=max(regrets),
                payoffs=list(map(float, payoffs)), best_responses=[float(a[0]) for a in best])


def solve_one(name, box, attempt, rules, output, presolve, source):
    full = name == "full"
    raw = solve_minimax(
        read_game(source), policy_mode="capture", tau=TAU, decision_margin=DELTA,
        seconds=600 if full else (60 if attempt == 1 else 120),
        round_seconds=90 if full else 30, threads=4 if full else 2,
        absolute_gap=0.001, max_rounds=40, strategy_bounds=box, initial_policies=rules,
        progress=None, presolve=presolve,
    )
    spec = read(ROOT / "data" / "refinement_rules.json")
    external_quote = spec["quote"] if full else spec["boxes"][name]["audit_quote"]
    checked = check_candidate(external_quote, source)
    raw["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    raw["external_checked_upper"] = checked["checked_upper"]
    if raw["raw_solver_lower_bound"] - DELTA > checked["checked_upper"]:
        raw["contradicted_master_lower"] = raw["raw_solver_lower_bound"]
        raw["lower_bound"] = 0.0
        raw["decision"] = "unresolved"
        raw["stop_reason"] = "contradicted_by_independent_feasible_quote"
    save(output / f"{name}_{attempt}.json", raw)
    print(name, attempt, raw["decision"], raw["lower_bound"], raw["upper_bound"],
          len(raw["history"]), flush=True)
    return raw


def trace(raw):
    return [dict(round=r["round"], lower=r["lower_bound"], upper=r["upper_bound"],
                 master_status=r["master_status"]) for r in raw["history"]]


def run(workers, output, resume, presolve=0, source=SOURCE):
    output.mkdir(exist_ok=True, parents=True)
    boxes = read(BOX_FILE)
    validate_cover(boxes)
    source = source.resolve()
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def batch(tasks):
            futures = {}
            for name, box, attempt, rules in tasks:
                path = output / f"{name}_{attempt}.json"
                if resume and path.is_file():
                    cached = read(path)
                    if cached.get("source_sha256") != fingerprint or cached["configuration"]["presolve"] != presolve:
                        raise ValueError(f"Cached input or solver settings mismatch: {path}")
                    results[name, attempt] = cached
                else:
                    future = pool.submit(solve_one, name, box, attempt, rules, output, presolve, source)
                    futures[future] = name, attempt
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        batch([(name, box, 1, None) for name, box in [("full", None), *boxes.items()]])
        full = results["full", 1]
        retry = []
        for name, box in boxes.items():
            first = results[name, 1]
            if first["decision"] == "unresolved":
                rules = [[*a, *(r for r in b if r not in a)] for a, b in
                         zip(first["deviation_policies"], full["deviation_policies"])]
                retry.append((name, box, 2, rules))
        batch(retry)
    leaves, rows = [], []
    for name, box in boxes.items():
        first = results[name, 1]
        final = results.get((name, 2), first)
        leaves.append(final)
        attempts = [first] if final is first else [first, final]
        rounds = [dict(r, round=i) for i, r in enumerate(
            (r for attempt in attempts for r in trace(attempt)), 1)]
        rows.append(dict(box=name, bounds=box, decision=final["decision"], stop=final["stop_reason"],
                         attempts=len(attempts), trace=rounds, lower=final["lower_bound"],
                         upper=final["upper_bound"]))
    lower = min(r["lower_bound"] if r["decision"] == "numerical_exclusion" else 0 for r in leaves)
    # The full-domain relaxation is also a valid global lower bound.
    lower = max(lower, full["lower_bound"])
    best_key, best = min(((k, r) for k, r in results.items() if r["best_candidate"] is not None),
                         key=lambda pair: pair[1]["upper_bound"])
    checked = check_candidate(best["best_candidate"]["x"], source)
    upper = checked["checked_upper"]
    decision = "unresolved"
    if lower <= upper and lower > TAU + DELTA:
        decision = "numerical_exclusion"
    elif lower <= upper <= TAU - DELTA:
        decision = "approximate_equilibrium"
    if lower > upper:
        lower = 0.0
    summary = dict(solver=coptpy.COPT_VERSION if hasattr(coptpy, "COPT_VERSION") else
                   str(coptpy.COPT.VERSION_MAJOR)+"."+str(coptpy.COPT.VERSION_MINOR)+"."+str(coptpy.COPT.VERSION_TECHNICAL),
                   python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__,
                   input_sha256=fingerprint,
                   tau=TAU, decision_margin=DELTA, workers=workers, presolve=presolve,
                   full_trace=trace(full), full_stop=full["stop_reason"], full_decision=full["decision"],
                   boxes=rows, lower=lower, decision=decision, candidate_source=best_key, **checked)
    save(output.parent / "results.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "physical/large/runs")
    parser.add_argument("--resume", action="store_true", help="reuse complete per-case JSONs in this output directory")
    parser.add_argument("--presolve", type=int, choices=[-1, 0], default=0)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    summary = run(args.workers, args.output, args.resume, args.presolve, args.source)
    print(summary["decision"], summary["lower"], summary["checked_upper"], flush=True)
