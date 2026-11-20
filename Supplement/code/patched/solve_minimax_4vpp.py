"""COPT implementation of the manuscript's rule-enhanced minimax-regret CCG.

Adapted from verification/ccg/solve_minimax_4vpp.py in MLMF_Figure_Code.zip.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import coptpy as cp
import numpy as np

from case20_game import MultiGame, SOURCE, load_game
from minimax_policies import best_response_policy


DEFAULT_OUTPUT = Path(__file__).with_suffix(".json")


# Validate candidate strategy box bounds while preserving the full unilateral deviation domain.
def validate_strategy_bounds(game: MultiGame, strategy_bounds=None) -> np.ndarray:
    """Normalize the candidate box; deviation domains are never restricted."""
    bounds = (
        np.tile([0.0, 1.0], (game.v, 1))
        if strategy_bounds is None
        else np.asarray(strategy_bounds, dtype=float)
    )
    if bounds.shape != (game.v, 2) or not np.all(np.isfinite(bounds)):
        raise ValueError("strategy_bounds must be a finite (vpp_count, 2) array")
    if np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > 1):
        raise ValueError("strategy_bounds must lie inside [0,1]")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("strategy_bounds must have lower < upper for every VPP")
    return bounds.copy()


# Validate and normalize one deviation rule independently of the deviating player strategy.
def validate_policy(game: MultiGame, player: int, policy):
    """Normalize saved affine or DER capture policies before model construction."""
    if isinstance(policy, dict):
        der = policy.get("der")
        offset = policy.get("offset", -1)
        if (
            isinstance(der, (bool, np.bool_))
            or not isinstance(der, (int, np.integer))
            or not 0 <= der < game.d
        ):
            raise ValueError("Capture policy der must be an integer within DER range")
        if isinstance(offset, (bool, np.bool_)) or offset not in (-1, 1):
            raise ValueError("Capture policy offset must be -1 or 1")
        kind = "capture" if offset == -1 else "departure"
        if policy.get("kind", kind) != kind:
            raise ValueError("Capture policy kind must agree with its offset")
        return {"kind": kind, "der": int(der), "offset": int(offset)}
    if not isinstance(policy, (tuple, list, np.ndarray)):
        raise ValueError("Deviation policy must be an affine sequence or capture dict")
    values = np.asarray(policy, dtype=float)
    if values.shape != (game.v + 1,) or not np.all(np.isfinite(values)):
        raise ValueError("Affine policy must contain vpp_count + 1 finite numbers")
    if values[player + 1] != 0:
        raise ValueError("Deviation policy must depend only on rivals")
    return tuple(map(float, values))


# Validate initial deviation rules and organize them into one collection per player.
def validate_initial_policies(game: MultiGame, initial_policies=None) -> list:
    """Accept the player-indexed deviation_policies field from a saved run."""
    if initial_policies is None:
        return [[] for _ in range(game.v)]
    if (
        not isinstance(initial_policies, (list, tuple))
        or len(initial_policies) != game.v
    ):
        raise ValueError("initial_policies must contain one policy list per VPP")
    normalized = []
    for player, policies in enumerate(initial_policies):
        if not isinstance(policies, (list, tuple)):
            raise ValueError(f"initial_policies[{player}] must be a policy list")
        normalized.append(
            [validate_policy(game, player, policy) for policy in policies]
        )
    return normalized


# Load the frozen game and validate coalition membership and compensation restrictions.
def read_game(source: Path) -> MultiGame:
    """Read the signed MATLAB model and validate its affiliation partition."""
    game, saved = load_game(source)
    counts = np.zeros(game.d, dtype=int)
    entries = np.atleast_1d(saved["cfg"]["vpp"])
    if len(entries) != game.v:
        raise ValueError("cfg.vpp count must match Parameter.FV columns")
    for entry in entries:
        members = np.atleast_1d(entry["members"])
        if not np.all(members == members.astype(int)):
            raise ValueError("cfg.vpp.members must contain integer indices")
        for member in members.astype(int):
            if not 1 <= member <= game.d:
                raise ValueError("cfg.vpp.members index outside DER range")
            counts[member - 1] += 1
    if not np.all(counts == 1):
        raise ValueError("cfg.vpp.members must partition all DERs exactly once")
    own = np.arange(game.v)[:, None] == game.incumbent
    if np.any(game.compensation < 0) or np.any(game.compensation[~own] != 0):
        raise ValueError("Require nonnegative compensation only for incumbents")
    return game


# Add follower response variables and KKT constraints without running the optimizer.
def add_follower_kkt(model: cp.Model, game: MultiGame, strategies, label: str):
    """Encode unique simplex responses, including all zero-share ties."""
    shares = []
    for der in range(game.d):
        y = model.addVars(game.v, lb=0.0, ub=1.0, nameprefix=f"{label}_y{der}")
        active = model.addVars(game.v, vtype=cp.COPT.BINARY, nameprefix=f"{label}_active{der}")
        water = model.addVar(
            lb=float(np.max(-game.c[der]) - game.epsilon),
            ub=float(np.max(game.f[der] - game.c[der])),
            name=f"{label}_lambda{der}",
        )
        model.addConstr(y.sum() == 1.0)
        for player in range(game.v):
            slack = (
                game.epsilon * y[player]
                + water
                - game.f[der, player] * (1.0 - strategies[player])
                + game.c[der, player]
            )
            model.addConstr(slack >= 0.0)
            model.addGenConstrIndicator(active[player], False, y[player] == 0.0)
            model.addGenConstrIndicator(active[player], True, slack == 0.0)
        shares.append(y)
    return shares


# Build a symbolic VPP payoff expression from its action and allocation variables.
def payoff_expression(game: MultiGame, player: int, action, shares):
    return action * game.base[player] + cp.quicksum(
        (action * game.contribution[player, der] - game.compensation[player, der])
        * shares[der][player]
        + game.compensation[player, der]
        for der in range(game.d)
    )


# Recompute candidate payoffs and complete numerical unilateral regrets independently of COPT.
def evaluate_candidate(game: MultiGame, strategies) -> dict:
    """Recompute a profile and every global unilateral best response."""
    x = np.asarray(strategies, dtype=float)
    if x.shape != (game.v,) or not np.all(np.isfinite(x)):
        raise ValueError("Candidate strategies have invalid shape or values")
    if np.any(x < -1e-7) or np.any(x > 1.0 + 1e-7):
        raise ValueError("Candidate strategy outside [0,1]")
    x = np.clip(x, 0.0, 1.0)
    profits = game.payoffs(x)
    responses = [game.best_response(i, x, value_tolerance=0.0) for i in range(game.v)]
    regrets = np.maximum(0.0, [responses[i].value - profits[i] for i in range(game.v)])
    return {
        "x": x.tolist(),
        "profits": profits.tolist(),
        "shares": game.shares(x).tolist(),
        "best_response_actions": [response.action for response in responses],
        "best_response_profits": [response.value for response in responses],
        "regrets": regrets.tolist(),
        "maximum_regret": float(np.max(regrets)),
    }


# Encode a clipped affine deviation action in the model without optimizing.
def add_policy(model, x, policy, label):
    """Encode a globally feasible clipped affine deviation rule."""
    a = policy[0]
    coefficients = np.asarray(policy[1:])
    if not np.any(coefficients):
        return float(np.clip(a, 0.0, 1.0))
    lower = float(a + np.minimum(coefficients, 0).sum())
    upper = float(a + np.maximum(coefficients, 0).sum())
    raw = model.addVar(lb=lower, ub=upper, name=f"{label}_raw")
    positive = model.addVar(lb=0.0, ub=max(0.0, upper), name=f"{label}_positive")
    action = model.addVar(lb=0.0, ub=1.0, name=f"{label}_action")
    model.addConstr(
        raw == a + cp.quicksum(b * x[j] for j, b in enumerate(coefficients))
    )
    model.addGenConstrMax(positive, [raw], constant=0.0)
    model.addGenConstrMin(action, [positive], constant=1.0)
    return action


# Evaluate a clipped capture or departure action against current rival offers.
def capture_action(game, player, der, x, offset=-1):
    if game.f[der, player] <= 0:
        return float("nan")
    rivals = np.delete(game.offers(x)[der], player)
    return float(
        np.clip(
            (
                game.f[der, player]
                - game.c[der, player]
                + offset * game.epsilon
                - max(rivals)
            )
            / game.f[der, player],
            0.0,
            1.0,
        )
    )


# Encode a clipped capture or departure rule using the highest rival offer.
def add_capture_policy(model, game, player, der, x, label, offset=-1):
    """A feasible quote tracking the maximum of all rival offers."""
    rivals = [j for j in range(game.v) if j != player]
    offers = []
    for j in rivals:
        offer = model.addVar(
            lb=-float(game.c[der, j]),
            ub=float(game.f[der, j] - game.c[der, j]),
            name=f"{label}_offer{j}",
        )
        model.addConstr(offer == game.f[der, j] * (1 - x[j]) - game.c[der, j])
        offers.append(offer)
    lower = float(max(-game.c[der, rivals]))
    upper = float(max(game.f[der, rivals] - game.c[der, rivals]))
    highest = model.addVar(lb=lower, ub=upper, name=f"{label}_highest")
    model.addGenConstrMax(highest, offers)
    intercept = game.f[der, player] - game.c[der, player] + offset * game.epsilon
    raw_lower = float((intercept - upper) / game.f[der, player])
    raw_upper = float((intercept - lower) / game.f[der, player])
    raw = model.addVar(lb=raw_lower, ub=raw_upper, name=f"{label}_raw")
    positive = model.addVar(lb=0, ub=max(0, raw_upper), name=f"{label}_positive")
    action = model.addVar(lb=0, ub=1, name=f"{label}_action")
    model.addConstr(raw == (intercept - highest) / game.f[der, player])
    model.addGenConstrMax(positive, [raw], constant=0.0)
    model.addGenConstrMin(action, [positive], constant=1.0)
    return action


# Save the current solver result as JSON and its available iteration history as CSV.
def save_result(result: dict, output: Path) -> None:
    """Persist partial progress so a timed run remains reviewable."""
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if result["history"]:
        with output.with_suffix(".csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["history"][0]))
            writer.writeheader()
            writer.writerows(result["history"])


# Optimize the minimax regret model with enhanced deviation rules and return numerical bounds and
# stopping status.
def solve_minimax(
    game: MultiGame,
    *,
    seconds=300.0,
    round_seconds=30.0,
    max_rounds=40,
    absolute_gap=1e-3,
    policy_mode="affine",
    tau=0.0,
    stop_on_verified=False,
    decision_margin=None,
    progress=print,
    log_path=None,
    checkpoint_path=None,
    initial=None,
    strategy_bounds=None,
    threads=4,
    initial_policies=None,
    presolve=None,
) -> dict:
    """Return minimax regret bounds without assuming equilibrium optimality.

    Constant policies implement ordinary deviation-scenario CCG. Affine policies
    also remain valid at every strategy: clipping ensures feasibility, and KKT
    recomputes their actual responses outside their generating support regions.
    Capture policies track the maximum rival offer and use the same KKT check.
    strategy_bounds restricts only candidate profiles. All best-response oracles
    and generated deviations still optimize over the original [0,1] domain.
    For a proper sub-box, returned bounds and decisions apply only to that box.
    Passing decision_margin selects Algorithm 1: tau is its regret tolerance.
    The midpoint only initializes the master. Apply both tests to master
    candidates and relax floating-point objective bounds by decision_margin.
    The default mode refines a minimax bracket to the requested absolute gap.
    """
    for name, value in (
        ("seconds", seconds),
        ("round_seconds", round_seconds),
        ("absolute_gap", absolute_gap),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if decision_margin is not None and not (0 <= decision_margin <= tau):
        raise ValueError("decision_margin must be between zero and tau")
    if max_rounds < 1 or policy_mode not in ("constant", "affine", "capture"):
        raise ValueError("Invalid max_rounds or policy_mode")
    if not math.isfinite(tau) or tau < 0 or np.any(game.compensation < 0):
        raise ValueError("tau and compensation must be nonnegative")
    if (
        isinstance(threads, (bool, np.bool_))
        or not isinstance(threads, (int, np.integer))
        or threads < 1
    ):
        raise ValueError("threads must be a positive integer")
    threads = int(threads)
    strategy_bounds = validate_strategy_bounds(game, strategy_bounds)
    initial_policies = validate_initial_policies(game, initial_policies)
    initial_values = np.asarray(
        np.full(game.v, 0.5) if initial is None else initial, dtype=float
    )
    if initial_values.shape != (game.v,) or not np.all(np.isfinite(initial_values)):
        raise ValueError("Initial strategies must contain one finite number per VPP")
    started = time.monotonic()
    best = evaluate_candidate(
        game, np.clip(initial_values, strategy_bounds[:, 0], strategy_bounds[:, 1])
    )
    lower_bound = 0.0
    raw_lower_bound = 0.0
    history = []
    policies = [[] for _ in range(game.v)]
    stop_reason = "round_limit"
    model_size = {}
    bounds = np.abs(game.base) + np.sum(
        np.abs(game.contribution) + game.compensation, axis=1
    )

    # Assemble current numerical bounds, decision status, and search scope into a result.
    def result():
        verified = decision_margin is None or bool(history)
        upper_bound = best["maximum_regret"] if verified else None
        if decision_margin is not None:
            inconsistent = verified and lower_bound > upper_bound
            reported_lower_bound = 0.0 if inconsistent else lower_bound
            decision = (stop_reason if stop_reason in ("approximate_equilibrium", "numerical_exclusion")
                        else "unresolved")
        else:
            margin = max(absolute_gap, 1e-6)
            if lower_bound > upper_bound + max(absolute_gap, 1e-5):
                raise RuntimeError(
                    "Global lower bound exceeds independent candidate upper bound"
                )
            reported_lower_bound = min(lower_bound, upper_bound)
            if reported_lower_bound > tau + margin:
                decision = "condition_5_numerically_verified"
            elif tau > 0 and upper_bound <= tau - margin:
                decision = "tau_equilibrium_numerically_verified"
            elif tau == 0 and upper_bound <= absolute_gap:
                decision = "approximate_equilibrium_candidate"
            else:
                decision = "undetermined"
            inconsistent = lower_bound > upper_bound
        return {
            "problem": "min_x max_i max_z [U_i(z,x_-i)-U_i(x)]",
            "vpp_count": game.v,
            "der_count": game.d,
            "epsilon": game.epsilon,
            "tau": tau,
            "policy_mode": policy_mode,
            "strategy_bounds": strategy_bounds.tolist(),
            "threads": threads,
            "verification_scope": (
                "full_strategy_domain"
                if np.all(strategy_bounds[:, 0] == 0)
                and np.all(strategy_bounds[:, 1] == 1)
                else "strategy_box"
            ),
            "configuration": {
                "seconds": seconds,
                "round_seconds": round_seconds,
                "max_rounds": max_rounds,
                "target_absolute_gap": absolute_gap,
                "stop_on_verified": stop_on_verified,
                "decision_margin": decision_margin,
                "strategy_bounds": strategy_bounds.tolist(),
                "threads": threads,
                "initial_policy_count": sum(map(len, initial_policies)),
                "presolve": presolve,
            },
            "stop_reason": stop_reason,
            "decision": decision,
            "lower_bound": reported_lower_bound,
            "raw_solver_lower_bound": raw_lower_bound,
            "upper_bound": upper_bound,
            "absolute_gap": max(0.0, upper_bound - reported_lower_bound) if verified else None,
            "bounds_overlap_within_tolerance": inconsistent,
            "exact_certificate": False,
            "certificate_type": "floating_point_global_master_and_piecewise_br_bounds",
            "elapsed_seconds": time.monotonic() - started,
            "best_candidate": best if verified else None,
            "deviation_policies": policies,
            "model_size": model_size,
            "history": history,
        }

    environment = cp.Envr()
    model = environment.createModel("multi_vpp_minimax_regret")
    for param, value in {"Logging": int(log_path is not None), "Threads": threads,
                         "NonConvex": 2, "FeasTol": 1e-9, "IntTol": 1e-9,
                         "DualTol": 1e-9, "RelGap": 0, "AbsGap": absolute_gap / 10}.items():
        model.setParam(param, value)
    if log_path is not None:
        model.setLogFile(str(Path(log_path).resolve()))
    if presolve is not None:
        model.setParam("Presolve", presolve)
    x = [model.addVar(lb=float(lo), ub=float(hi), name=f"x({i})")
         for i, (lo, hi) in enumerate(strategy_bounds)]
    regret = model.addVar(lb=0.0, ub=float(np.max(bounds)), name="regret")
    y = add_follower_kkt(model, game, x, "current")
    profit = [model.addVar(lb=-float(bound), ub=float(bound), name=f"profit({i})")
              for i, bound in enumerate(bounds)]
    for i in range(game.v):
        model.addQConstr(profit[i] == payoff_expression(game, i, x[i], y))
    model.setObjective(regret, cp.COPT.MINIMIZE)

    # Add one distinct deviation rule and its own follower response block without
    # optimizing.
    def add_deviation(player, policy):
        policy = validate_policy(game, player, policy)
        if isinstance(policy, dict):
            if policy in policies[player]:
                return False
        else:
            if any(
                not isinstance(old, dict)
                and np.max(np.abs(np.subtract(policy, old))) < 1e-12
                for old in policies[player]
            ):
                return False
        label = f"dev{player}_{len(policies[player])}"
        if isinstance(policy, dict):
            action = add_capture_policy(
                model,
                game,
                player,
                policy["der"],
                x,
                label,
                policy.get("offset", -1),
            )
        else:
            action = add_policy(model, x, policy, label)
        profile = [x[j] for j in range(game.v)]
        profile[player] = action
        response = add_follower_kkt(model, game, profile, label)
        add_gain = model.addConstr if isinstance(action, float) else model.addQConstr
        add_gain(
            regret + profit[player]
            >= payoff_expression(game, player, action, response),
            name=f"{label}_gain",
        )
        policies[player].append(policy)
        return True

    # Convert candidate best responses into additional affine, constant, capture, or
    # departure rules.
    def add_candidate_deviations(candidate):
        added = 0
        for player, action in enumerate(candidate["best_response_actions"]):
            policy = (action, *np.zeros(game.v))
            if policy_mode in ("affine", "capture"):
                policy = best_response_policy(
                    game, player, candidate["x"], action
                )
            if policy_mode == "capture" and 0 < action < 1:
                matches = [
                    {
                        "kind": "capture" if offset == -1 else "departure",
                        "der": der,
                        "offset": offset,
                    }
                    for offset in (-1, 1)
                    for der in range(game.d)
                    if abs(
                        capture_action(
                            game, player, der, candidate["x"], offset
                        )
                        - action
                    )
                    < 1e-10
                ]
                if matches:
                    policy = matches[0]
            added += add_deviation(player, policy)
        return added

    for i in range(game.v):
        for action in (0.0, 1.0):
            add_deviation(i, (action, *np.zeros(game.v)))
        for policy in initial_policies[i]:
            add_deviation(i, policy)
    if initial is not None:
        add_candidate_deviations(best)

    # Populate model start values consistently for the candidate and every encoded deviation
    # rule.
    def set_complete_start(candidate):
        """Supply every original binary/continuous response variable."""
        starts = {}
        variables = {variable.name: variable for variable in model.getVars()}
        values = np.asarray(candidate["x"])
        for i in range(game.v):
            starts[x[i].name] = values[i]
            starts[profit[i].name] = candidate["profits"][i]
        starts[regret.name] = candidate["maximum_regret"]
        profiles = [("current", values)]
        for player, rules in enumerate(policies):
            for index, policy in enumerate(rules):
                label = f"dev{player}_{index}"
                profile = values.copy()
                if isinstance(policy, dict):
                    der = policy["der"]
                    rivals = [j for j in range(game.v) if j != player]
                    offers = game.offers(values)[der]
                    for j in rivals:
                        starts[f"{label}_offer{j}"] = offers[j]
                    highest = float(max(offers[rivals]))
                    starts[f"{label}_highest"] = highest
                    raw = (
                        game.f[der, player]
                        - game.c[der, player]
                        + policy.get("offset", -1) * game.epsilon
                        - highest
                    ) / game.f[der, player]
                else:
                    raw = policy[0] + np.dot(policy[1:], values)
                profile[player] = np.clip(raw, 0, 1)
                if f"{label}_raw" in variables:
                    starts[f"{label}_raw"] = raw
                    starts[f"{label}_positive"] = max(0, raw)
                    starts[f"{label}_action"] = profile[player]
                profiles.append((label, profile))
        for label, profile in profiles:
            response = game.shares(profile)
            offers = game.offers(profile)
            for der in range(game.d):
                largest = int(np.argmax(offers[der]))
                water = (
                    offers[der, largest] - game.epsilon * response[der, largest]
                )
                starts[f"{label}_lambda{der}"] = water
                for i in range(game.v):
                    starts[f"{label}_y{der}({i})"] = response[der, i]
                    starts[f"{label}_active{der}({i})"] = int(
                        response[der, i] > 0
                    )

        model.setMipStart([variables[name] for name in starts], list(starts.values()))
        model.loadMipStart()

    refresh_start = True
    for iteration in range(1, max_rounds + 1):
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0:
            stop_reason = "time_limit"
            break
        model.setParam("TimeLimit", min(round_seconds, remaining))
        if refresh_start:
            set_complete_start(best)
        model.solve()
        if model.status not in (cp.COPT.OPTIMAL, cp.COPT.TIMEOUT):
            if decision_margin is not None:
                stop_reason = f"master_status_{model.status}"
                break
            raise RuntimeError(
                f"No reliable master bound; solver status {model.status}"
            )
        if math.isfinite(model.bestbnd):
            raw_lower_bound = max(raw_lower_bound, float(model.bestbnd))
            lower_bound = max(0.0, raw_lower_bound - (decision_margin or 0.0))
        model_size = {
            "variables": model.cols,
            "binary_variables": model.bins,
            "linear_constraints": model.rows,
            "quadratic_constraints": model.qconstrs,
            "general_constraints": len(model.getGenConstrs()) + model.indicators,
        }
        if not model.hasmipsol:
            stop_reason = "no_master_incumbent"
            break
        master_x = np.array([variable.x for variable in x])
        clipped = np.clip(master_x, strategy_bounds[:, 0], strategy_bounds[:, 1])
        if np.max(np.abs(master_x - clipped)) > 1e-7:
            raise ValueError(f"Master candidate outside its strategy box: {master_x}")
        candidate = evaluate_candidate(game, clipped)
        first_master_candidate = decision_margin is not None and not history
        if first_master_candidate or candidate["maximum_regret"] < best["maximum_regret"]:
            best = candidate
        if decision_margin is None and lower_bound > best["maximum_regret"] + max(absolute_gap, 1e-5):
            raise RuntimeError(
                "Global lower bound exceeds independent candidate upper bound"
            )
        gap = max(0.0, best["maximum_regret"] - lower_bound)
        history.append(
            {
                "round": iteration,
                "master_status": int(model.status),
                "master_objective": float(model.bestobj),
                "lower_bound": lower_bound,
                "raw_solver_lower_bound": raw_lower_bound,
                "upper_bound": best["maximum_regret"],
                "absolute_gap": gap,
                "elapsed_seconds": time.monotonic() - started,
                "policies": sum(map(len, policies)),
                **{
                    f"x{i + 1}": value for i, value in enumerate(candidate["x"])
                },
                **{
                    f"regret{i + 1}": value
                    for i, value in enumerate(candidate["regrets"])
                },
            }
        )
        if progress:
            progress(
                f"round {iteration:02d}: LB={lower_bound:.9f} "
                f"UB={best['maximum_regret']:.9f} gap={gap:.6g} "
                f"elapsed={time.monotonic() - started:.1f}s "
                f"x={np.round(candidate['x'], 9).tolist()}",
                flush=True,
            )
        if checkpoint_path is not None:
            save_result(result(), Path(checkpoint_path))
        if decision_margin is not None:
            if lower_bound > best["maximum_regret"]:
                stop_reason = "inconsistent_bounds"
            elif best["maximum_regret"] <= tau - decision_margin:
                stop_reason = "approximate_equilibrium"
            elif lower_bound > tau + decision_margin:
                stop_reason = "numerical_exclusion"
            if stop_reason != "round_limit":
                break
        elif (
            stop_on_verified
            and result()["decision"] == "condition_5_numerically_verified"
        ):
            stop_reason = "condition_5_verified"
            break
        if decision_margin is None and gap <= absolute_gap:
            stop_reason = "gap_reached"
            break
        added = add_candidate_deviations(candidate)
        refresh_start = bool(added)
        if not added and (decision_margin is not None or model.status == cp.COPT.OPTIMAL):
            stop_reason = "separation_stalled_numerically"
            break

    final = result()
    if checkpoint_path is not None:
        save_result(final, Path(checkpoint_path))
    return final


