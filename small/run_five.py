"""COPT replication of the five-DER contract controls and baseline CCG."""
from fractions import Fraction as F
from pathlib import Path
import json
import sys
import coptpy as cp
import numpy as np

from exact_audit import FrozenGame, full_regrets, quadratic_from_three

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "large" / "core"))
from case20_game import MultiGame
from solve_minimax_4vpp import solve_minimax


def response(env, game, player, rival=F(1)):
    """Optimize each smooth piece as a normalized continuous COPT quadratic program."""
    breaks = {F(0), F(1)}
    for line in game.difference:
        intercept = line[0] + line[2-player] * rival
        for threshold in (-game.epsilon, game.epsilon):
            quote = (threshold - intercept) / line[1+player]
            if 0 < quote < 1:
                breaks.add(quote)

    def payoff(own):
        profile = (own, rival) if player == 0 else (rival, own)
        return game.payoff(player, profile)

    best, best_quote = None, None
    points = sorted(breaks)
    for left, right in zip(points, points[1:]):
        _, linear, quad = quadratic_from_three(payoff(left), payoff((left+right)/2), payoff(right))
        model = env.createModel()
        model.setParam("Logging", 0)
        model.setParam("Threads", 1)
        model.setParam("FeasTol", 1e-9)
        model.setParam("DualTol", 1e-9)
        # For a convex polynomial its maximum is at an endpoint.
        if quad > 0:
            t = model.addVar(vtype=cp.COPT.BINARY)
            model.setObjective(float(linear + quad) * t, cp.COPT.MAXIMIZE)
        else:
            t = model.addVar(lb=0, ub=1)
            model.setObjective(float(linear) * t + float(quad) * t*t, cp.COPT.MAXIMIZE)
        model.solve()
        if model.status != cp.COPT.OPTIMAL:
            raise RuntimeError(f"COPT best-response status {model.status}")
        quote = left + (right - left) * F(min(1., max(0., t.x)))
        value = payoff(quote)
        if best is None or value > best:
            best, best_quote = value, quote
    exact = game.best_response_value(player, rival)
    if abs(float(best - exact)) > 1e-5:
        raise ArithmeticError("COPT best response differs from rational global maximum")
    return dict(action=float(best_quote), regret=float(max(F(0), best - payoff(F(1)))),
                exact_regret=float(exact - payoff(F(1))), intervals_solved=len(breaks)-1)


def audit_trace(raw, game, use_midpoint):
    rows = []
    best_quote = None
    upper = float("inf")
    if use_midpoint:
        best_quote = [0.5, 0.5]
        upper = float(max(full_regrets(game, (F(1,2), F(1,2)))))
        rows.append(dict(round=0, lower=0., upper=upper, candidate=best_quote))
    for step in raw["history"]:
        quote = [step["x1"], step["x2"]]
        regrets = list(map(float, full_regrets(game, tuple(map(F, quote)))))
        if max(regrets) < upper:
            upper, best_quote = max(regrets), quote
        lower = max(0., step["raw_solver_lower_bound"] - 1e-6)
        if lower > upper:
            raise ArithmeticError("Global lower bound exceeds exact candidate regret")
        rows.append(dict(round=step["round"], candidate=quote, lower=lower, upper=upper,
                         regrets=regrets, raw_lower=step["raw_solver_lower_bound"], policy_count=step["policies"]))
    return rows, best_quote


