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


def controls(baseline):
    env = cp.Envr()
    witnesses = json.loads((HERE / "inputs" / "expiry_witnesses.json").read_bytes())
    rows = []
    for state in baseline["protected_states"]:
        game = FrozenGame(state["inputs"])
        responses = [response(env, game, i) for i in range(2)]
        gains = [item["regret"] for item in responses]
        mask = {"None":"0", "ESS":"8", "All":"31"}[state["expired"]]
        checked = []
        for quote in witnesses.get(mask, []):
            regrets = full_regrets(game, tuple(map(F, quote)))
            if max(regrets) != 0:
                raise ArithmeticError("Invalid archived exact equilibrium witness")
            checked.append(dict(quote=quote, regrets=list(map(float, regrets))))
        rows.append(dict(expired=state["expired"], regrets=gains, responses=responses,
                         exact_equilibria_checked=checked))
    return rows


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


def main():
    baseline = json.loads((HERE / "inputs" / "baseline.json").read_bytes())
    protected = controls(baseline)
    inputs = baseline["inputs"]
    data = {k: np.asarray(v) for k,v in inputs.items() if k != "epsilon"}
    game = MultiGame(**data, epsilon=inputs["epsilon"])
    exact = FrozenGame(inputs)
    settings = baseline["ccg_settings"]
    decision_settings = {**settings, "tau": 0.001, "decision_margin": 1e-6}
    decision = solve_minimax(game, **decision_settings)
    decision_trace, _ = audit_trace(decision, exact, False)
    refinement = solve_minimax(game, **settings)
    trace, quote = audit_trace(refinement, exact, True)
    out = dict(protected_states=protected, trace=trace,
               settings=settings, decision_settings=decision_settings, decision_trace=decision_trace,
               decision=decision["decision"], best_quote=quote,
               exact_audited_regrets=list(map(float, full_regrets(exact, tuple(map(F, quote))))),
               first_exclusion_round=next((s["round"] for s in decision_trace if s["lower"] > 0.001001),None),
               refinement_gap=trace[-1]["upper"] - trace[-1]["lower"], refinement_stop=refinement["stop_reason"])
    (HERE / "results.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

