"""Native COPT first-return driver for two-leader control and expiry sweeps."""
import argparse
import hashlib
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import coptpy as cp
import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "patched"))
from case20_game import MultiGame
from solve_minimax_4vpp import solve_minimax
from verify_cases import FrozenGame, full_regrets

ETA = 1e-3
DELTA = 1e-6


def read(name):
    return json.loads((HERE / name).read_bytes())


def settings(case):
    large = case == "twenty"
    control = case == "control"
    return dict(tau=2e-5 if control else ETA, decision_margin=DELTA,
                policy_mode="capture", seconds=600 if large else 300,
                round_seconds=90 if large else 30, max_rounds=40 if large else 200,
                absolute_gap=2e-5 if control else 1e-3, threads=4 if large or control else 2)


def numeric_game(inputs):
    arrays = {name: np.asarray(inputs[name], dtype=float)
              for name in ("f", "c", "base", "contribution", "compensation")}
    return MultiGame(**arrays, epsilon=inputs["epsilon"],
                     incumbent=np.asarray(inputs["incumbent"], dtype=int))


def expiry_inputs(protected, mask):
    # Work from the stored full-precision protected coefficients.
    deductions = [list(row) for row in protected["c"]]
    fees = [list(row) for row in protected["compensation"]]
    for d, owner in enumerate(protected["incumbent"]):
        if mask & (1 << d):
            deductions[d] = [0.0, 0.0]
            fees[owner][d] = 0.0
    return {**protected, "c": deductions, "compensation": fees}


def summarize_result(raw, key, players):
    trace = []
    for step in raw["history"]:
        candidate = [step[f"x{i + 1}"] for i in range(players)]
        trace.append(dict(round=step["round"], lower=step["lower_bound"],
                          upper=step["upper_bound"], candidate=candidate,
                          raw_bound=step["raw_solver_lower_bound"], rules=step["policies"]))
    best = raw["best_candidate"]
    return dict(key=key, decision=raw["decision"], stop=raw["stop_reason"], trace=trace,
                lower=raw["lower_bound"], upper=raw["upper_bound"],
                quote=best["x"] if best else None)


def two_player_run(task, presolve=None):
    case, key, inputs, witnesses = task
    raw = solve_minimax(numeric_game(inputs), progress=None, presolve=presolve, **settings(case))
    row = summarize_result(raw, key, 2)
    exact = FrozenGame(inputs)
    if row["quote"] is not None:
        point = tuple(map(Fraction, row["quote"]))
        regrets = full_regrets(exact, point)
        row["checked_upper"] = float(max(regrets))
        row["regrets"] = list(map(float, regrets))
        eta = settings(case)["tau"]
        if (row["lower"] > max(regrets) or
                (row["decision"] == "approximate_equilibrium" and max(regrets) > eta - DELTA)):
            row.update(decision="unresolved", stop="independent_bound_check")
    # Exact witnesses are used only after the CCG run and never initialize its master.
    for witness in witnesses:
        point = tuple(map(Fraction, witness))
        if not all(0 <= x <= 1 for x in point) or max(full_regrets(exact, point)) != 0:
            raise ValueError(f"Invalid equilibrium witness for {key}: {witness}")
    if witnesses and row["lower"] > 0:
        row.update(decision="unresolved", stop="exact_witness_contradicts_bound")
    row["verified_exact_profiles"] = len(witnesses)
    if case in ("five", "control", "expiry"):
        row["model_size"] = raw["model_size"]
        row["elapsed_seconds"] = raw["elapsed_seconds"]
    return row


