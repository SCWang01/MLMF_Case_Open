"""Propagate COPT five-DER dispatch incumbents through the paper's allocation rules and CCG."""
import csv
from fractions import Fraction
from itertools import combinations
from math import comb
import json
from pathlib import Path
import sys

import coptpy as cp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "small"))
from run_five import FrozenGame, MultiGame, audit_trace, full_regrets, response, solve_minimax


def coefficients(records):
    value = {frozenset():0.}
    for key, val in records.items():
        if key:
            value[frozenset(map(int, key.split(",")))] = val
    surplus = {s: v - sum(value[frozenset([d])] for d in s) for s,v in value.items()}
    current = [frozenset((1,3,4)), frozenset((2,5))]
    base, f, contribution = np.empty(2), np.empty((5,2)), np.empty((2,5))
    for i, coalition in enumerate(current):
        for d in range(1,6):
            enlarged = coalition | {d}
            marginal = {j:surplus[enlarged] - surplus[enlarged - {j}] for j in enlarged}
            contribution[i, d-1] = surplus[enlarged] - surplus[enlarged - {d}]
            if i == 0:
                f[d-1,i] = surplus[enlarged]*marginal[d]/sum(marginal.values())
            else:
                rest = enlarged - {d}
                total = 0.
                for size in range(len(rest)+1):
                    for subset in combinations(rest,size):
                        sub = frozenset(subset)
                        total += (surplus[sub | {d}] - surplus[sub]) / (len(enlarged)*comb(len(rest),size))
                f[d-1,i] = total
        base[i] = surplus[coalition] - sum(contribution[i,d-1] for d in coalition)
    return dict(base=base.tolist(), f=f.tolist(), contribution=contribution.tolist())


def main():
    with (HERE / "five_tight.csv").open(encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if len(rows) != 31 or any(row["optimal_within_tolerance"] != "True" for row in rows):
        raise ValueError("Expected a solved table of 31 nonempty coalitions")
    ref = coefficients({r["members"]:float(r["paper_input_profit"]) for r in rows})
    rebuilt = coefficients({r["members"]:float(r["lower"]) for r in rows})
    baseline = json.loads((HERE.parent / "small/inputs/baseline.json").read_bytes())
    delta = max(abs(np.asarray(ref[k]) - baseline["inputs"][k]).max() for k in ref)
    if delta > 1e-8:
        raise ArithmeticError("Archived coalition table does not recover manuscript game")
    inputs = dict(baseline["inputs"], **rebuilt)
    exact = FrozenGame(inputs)
    game = MultiGame(**{k:np.array(v) if k != "epsilon" else v for k,v in inputs.items()})
    settings = baseline["ccg_settings"]
    decision = solve_minimax(game, **{**settings,"tau":0.001,"decision_margin":1e-6})
    decision_trace, _ = audit_trace(decision, exact, False)
    result = solve_minimax(game, **settings)
    trace, quote = audit_trace(result, exact, True)
    env = cp.Envr()
    states = []
    for state in baseline["protected_states"]:
        game_state = FrozenGame(dict(state["inputs"], **rebuilt))
        responses = [response(env, game_state, i) for i in range(2)]
        row = dict(expired=state["expired"], regrets=[r["regret"] for r in responses])
        if state["expired"] == "All":
            # At rival quote zero, try the binding ESS support boundary and
            # independently check both players across their full quote domains.
            intercept, slope, _ = game_state.difference[3]
            ess_boundary = (game_state.epsilon-intercept)/slope
            profile = (ess_boundary, Fraction(0))
            regrets = full_regrets(game_state,profile)
            if 0 <= ess_boundary <= 1 and max(regrets) == 0:
                row["exact_equilibrium"] = [str(v) for v in profile]
                row["exact_equilibrium_regrets"] = [str(v) for v in regrets]
        states.append(row)
    out = dict(input_basis="COPT feasible coalition dispatch objectives from five_tight.csv", inputs=inputs,
               reference_coefficient_reconstruction_error=delta,
               maximum_dispatch_gap=max(float(r["upper"])-float(r["lower"]) for r in rows),
               coefficient_max_differences={k:float(np.max(np.abs(np.asarray(rebuilt[k])-ref[k]))) for k in ref},
               decision=decision["decision"], decision_trace=decision_trace, trace=trace, best_quote=quote,
               first_exclusion_round=next((s["round"] for s in decision_trace if s["lower"]>0.001001),None),
               refinement_stop=result["stop_reason"],
               exact_audited_regrets=list(map(float, full_regrets(exact, tuple(map(Fraction, quote))))),
               protected_states=states)
    (HERE/"five_end_to_end.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n", encoding="utf-8")
    print(quote, trace[-1], states)


if __name__ == "__main__":
    main()

