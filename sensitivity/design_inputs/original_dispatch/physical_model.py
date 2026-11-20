"""Reproduce the saved VPP physical model on 24 quarter-hour intervals.

The saved five-DER MAT supplies every scenario and numerical coefficient.
Legacy per-interval prices/costs and MW-quarter-hour capacities are converted
to hourly rates and MWh before explicit multiplication by dt=0.25. Only the
incumbent ESS power limit changes in the controlled experiment.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import gurobipy as gp
import numpy as np
from scipy.io import loadmat

SOURCE = Path(__file__).resolve().parents[3] / "physical/source/five/5DER_electricity.mat"
DT = 0.25
MODEL_ID = "original_physical_6h_24x15min_explicit_units_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_parameters() -> dict:
    """Copy authoritative five-DER values, retaining both unit representations."""
    legacy = loadmat(SOURCE, simplify_cells=True)["Parameter"]
    p = {name: np.asarray(value, dtype=float) for name, value in legacy.items()}
    if p["Lmpda"].shape != (24, 18) or not np.isclose(p["P"].sum(), 1.0):
        raise ValueError("Unexpected source dimensions or scenario probabilities")
    for name, value in p.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Nonfinite source parameter: {name}")
    return p


def canonical_assets(assets: list[dict]) -> list[dict]:
    """Discard interchangeable labels while retaining every physical parameter."""
    return sorted(
        [{"type": a["type"], "power_fraction": float(a.get("power_fraction", 1.0))} for a in assets],
        key=lambda a: (a["type"], a["power_fraction"]),
    )


def input_identity(assets: list[dict], settlement: str = "original") -> dict:
    return {"model_id": MODEL_ID, "source_sha256": sha256(SOURCE),
            "implementation_sha256": sha256(Path(__file__)), "dt_hours": DT,
            "periods": 24, "scenario_count": 18, "settlement": settlement,
            "assets": canonical_assets(assets)}


def cache_key(assets: list[dict], settlement: str = "original") -> str:
    content = json.dumps(input_identity(assets, settlement), sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def solve_coalition(assets: list[dict], output: Path, seconds: float = 120,
                    threads: int = 4, fixed_bid=None, settlement: str = "original") -> dict:
    """Solve one convex MIQP and save dispatch, objective interval and audits.

    fixed_bid holds the saved day-ahead bid when auditing historical recourse.
    Original settlement preserves aggregate net deviations and the original
    80%-130% delivery rule. The additive control is assembled by the driver
    from separately settled singleton problems, preserving local constraints.
    """
    if settlement != "original":
        raise ValueError("Unknown settlement")
    assets = canonical_assets(assets)
    if any(a["type"] not in {"WT", "PV", "TG", "ESS", "DR"} or
           not 0 < a["power_fraction"] <= 1 for a in assets):
        raise ValueError("Invalid physical asset or power fraction")
    output.mkdir(parents=True, exist_ok=True)
    identity = input_identity(assets, settlement)
    if fixed_bid is not None:
        identity["fixed_day_ahead_bid"] = np.asarray(fixed_bid, dtype=float).tolist()
    if not assets:
        result = {**identity, "status": "analytic_empty_coalition", "lower_bound": 0.,
                  "upper_bound": 0., "absolute_gap": 0., "elapsed_seconds": 0.,
                  "physical_audit": {"maximum_violation": 0.}, "optimization_executed": False}
        write_json(output / "result.json", result)
        return result
    p = load_parameters()
    periods, scenarios = p["Lmpda"].shape
    price_da = p["Lmpda"] / DT
    price_rt = p["Lmprt"] / DT
    penalty_rate = float(p["pane"]) / DT
    probability = p["P"]
    started = time.monotonic()
    arrays = {}
    blocks = []
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("physical_6h", env=env) as model:
            model.Params.Threads = threads
            model.Params.Seed = 0
            model.Params.TimeLimit = seconds
            model.Params.MIPGap = 1e-6
            model.Params.MIPGapAbs = 1e-6
            model.Params.FeasibilityTol = 1e-8
            model.Params.IntFeasTol = 1e-8
            model.Params.OptimalityTol = 1e-8
            model.Params.OutputFlag = 1
            model.Params.LogToConsole = 0
            model.Params.LogFile = str(output / "solver.log")

            def resource(asset: dict, index: int, stage: str):
                count = 1 if stage == "da" else scenarios
                tag = f"{stage}_{index}_{asset['type']}"
                kind = asset["type"]
                shape = (periods, count)
                cost = [[gp.QuadExpr() for _ in range(count)] for _ in range(periods)]
                variables = {}
                if kind in {"WT", "PV"}:
                    actual = p["Wind_f" if kind == "WT" else "Solar_f"]
                    lower = actual.min(axis=1)[:, None] if stage == "da" else actual
                    upper = actual.max(axis=1)[:, None] if stage == "da" else actual
                    power = model.addVars(periods, count, lb=0, name=tag + "_power")
                    for t in range(periods):
                        for w in range(count):
                            power[t, w].LB = float(lower[t, w])
                            power[t, w].UB = float(upper[t, w])
                    variables["power"] = power
                    net = power
                elif kind == "TG":
                    power = model.addVars(periods, count, lb=0, name=tag + "_power")
                    on = model.addVars(periods, count, vtype=gp.GRB.BINARY, name=tag + "_on")
                    for t in range(periods):
                        for w in range(count):
                            model.addConstr(power[t, w] >= float(p["Pgmin"]) * on[t, w])
                            model.addConstr(power[t, w] <= float(p["Pgmax"]) * on[t, w])
                            if t == 0:
                                model.addConstr(on[t, w] == 0)
                            else:
                                model.addConstr(power[t, w] - power[t-1, w] <= (float(p["Ug"]) / DT) * DT)
                                model.addConstr(power[t-1, w] - power[t, w] <= (float(p["Dg"]) / DT) * DT)
                            cost[t][w] = DT * ((float(p["ag"]) / DT) * power[t, w]
                                             + (float(p["cg"]) / DT) * power[t, w] * power[t, w]
                                             + (float(p["bg"]) / DT) * on[t, w])
                    variables = {"power": power, "on": on}
                    net = power
                elif kind == "ESS":
                    fraction = asset["power_fraction"]
                    charge = model.addVars(periods, count, lb=0, name=tag + "_charge")
                    discharge = model.addVars(periods, count, lb=0, name=tag + "_discharge")
                    charging = model.addVars(periods, count, vtype=gp.GRB.BINARY, name=tag + "_charging")
                    discharging = model.addVars(periods, count, vtype=gp.GRB.BINARY, name=tag + "_discharging")
                    soc = model.addVars(periods, count, lb=float(p["socmin"]), ub=float(p["socmax"]), name=tag + "_soc")
                    capacity_mwh = float(p["Esmax"]) * DT
                    net = {}
                    for t in range(periods):
                        for w in range(count):
                            model.addConstr(charge[t, w] <= float(p["p_cmax"]) * fraction * charging[t, w])
                            model.addConstr(discharge[t, w] <= float(p["p_dmax"]) * fraction * discharging[t, w])
                            model.addConstr(charging[t, w] + discharging[t, w] <= 1)
                            previous = 0.5 if t == 0 else soc[t-1, w]
                            model.addConstr(soc[t, w] == previous + DT / capacity_mwh *
                                            (float(p["yt_sin"]) * charge[t, w] - discharge[t, w] / float(p["yt_sout"])))
                            if t == periods - 1:
                                model.addConstr(soc[t, w] == 0.5)
                            net[t, w] = discharge[t, w] - charge[t, w]
                            cost[t][w] = DT * (float(p["as"]) / DT) * (charge[t, w] + discharge[t, w])
                    variables = {"charge": charge, "discharge": discharge, "charging": charging,
                                 "discharging": discharging, "soc": soc}
                else:
                    limit = p["Load_base"][:, None] if stage == "da" else p["Load_f"]
                    power = model.addVars(periods, count, lb=0, name=tag + "_power")
                    for t in range(periods):
                        for w in range(count):
                            power[t, w].UB = 0.85 * float(limit[t, w])
                            cost[t][w] = DT * (float(p["Cutting"]) / DT) * power[t, w]
                    variables = {"power": power}
                    net = power
                blocks.append((asset, index, stage, variables, count))
                return net, cost

            da = [resource(asset, i, "da") for i, asset in enumerate(assets)]
            rt = [resource(asset, i, "rt") for i, asset in enumerate(assets)]
            bid = model.addVars(periods, lb=0, name="day_ahead_bid")
            delivered = model.addVars(periods, scenarios, lb=0, name="delivered")
            active = model.addVars(periods, scenarios, vtype=gp.GRB.BINARY, name="delivery_active")
            absolute_deviation = model.addVars(periods, scenarios, lb=0, name="absolute_deviation")
            objective = gp.QuadExpr()
            for t in range(periods):
                model.addConstr(bid[t] == gp.quicksum(net[t, 0] for net, _ in da))
                if fixed_bid is not None:
                    model.addConstr(bid[t] == float(fixed_bid[t]))
                for w in range(scenarios):
                    model.addConstr(delivered[t, w] <= gp.quicksum(net[t, w] for net, _ in rt))
                    model.addConstr(delivered[t, w] <= 1.3 * bid[t])
                    model.addGenConstrIndicator(active[t, w], False, delivered[t, w] == 0)
                    model.addGenConstrIndicator(active[t, w], True, delivered[t, w] >= 0.8 * bid[t])
                    deviation = delivered[t, w] - bid[t]
                    model.addConstr(absolute_deviation[t, w] >= deviation)
                    model.addConstr(absolute_deviation[t, w] >= -deviation)
                    revenue = DT * (float(price_da[t, w]) * bid[t] + float(price_rt[t, w]) * deviation
                                    - penalty_rate * absolute_deviation[t, w])
                    costs = gp.quicksum(cost[t][w] for _, cost in rt)
                    objective += float(probability[w]) * (revenue - costs)
            model.setObjective(objective, gp.GRB.MAXIMIZE)
            model.optimize()
            result = {**identity, "optimization_executed": True, "solver_status": int(model.Status),
                      "solution_count": model.SolCount, "threads": threads, "time_limit": seconds,
                      "solver_version": list(gp.gurobi.version()), "solver_runtime": model.Runtime,
                      "mip_gap_target": 1e-6, "absolute_gap_target": 1e-6,
                      "variables": model.NumVars, "binary_variables": model.NumBinVars,
                      "constraints": model.NumConstrs, "general_constraints": model.NumGenConstrs,
                      "elapsed_seconds": time.monotonic() - started}
            if model.SolCount:
                arrays["bid"] = np.array([bid[t].X for t in range(periods)])
                for name, group in (("delivered", delivered), ("delivery_active", active), ("absolute_deviation", absolute_deviation)):
                    arrays[name] = np.array([[group[t, w].X for w in range(scenarios)] for t in range(periods)])
                for asset, index, stage, variables, count in blocks:
                    for name, group in variables.items():
                        arrays[f"{stage}_{index}_{asset['type']}_{name}"] = np.array(
                            [[group[t, w].X for w in range(count)] for t in range(periods)])
                np.savez_compressed(output / "dispatch.npz", **arrays)
                audit = audit_dispatch(p, assets, arrays)
                audit["objective_reconstruction_error"] = abs(audit["reconstructed_objective"] - float(model.ObjVal))
                result.update(lower_bound=float(model.ObjVal), upper_bound=float(model.ObjBound),
                              absolute_gap=max(0., float(model.ObjBound-model.ObjVal)),
                              physical_audit=audit, gurobi_constraint_violation=float(model.ConstrVio),
                              gurobi_integrality_violation=float(model.IntVio))
                result["status"] = "validated_numeric_optimum" if model.Status == gp.GRB.OPTIMAL and audit["maximum_violation"] <= 1e-6 and audit["objective_reconstruction_error"] <= 1e-5 else "bounded_numeric_result"
            else:
                result["status"] = "no_feasible_solution"
            write_json(output / "result.json", result)
            return result


def audit_dispatch(p: dict, assets: list[dict], arrays: dict) -> dict:
    """Check physical equations independently from saved numerical dispatch."""
    violations = {}
    costs = np.zeros_like(p["Lmpda"])
    supplies = {}

    def check(name, residual):
        violations[name] = max(violations.get(name, 0.), float(np.max(np.maximum(0., residual))))

    for stage in ("da", "rt"):
        supply = np.zeros((24, 1 if stage == "da" else 18))
        for i, asset in enumerate(assets):
            kind = asset["type"]
            prefix = f"{stage}_{i}_{kind}_"
            if kind in ("WT", "PV"):
                power = arrays[prefix + "power"]
                actual = p["Wind_f" if kind == "WT" else "Solar_f"]
                low = actual.min(axis=1)[:, None] if stage == "da" else actual
                high = actual.max(axis=1)[:, None] if stage == "da" else actual
                check("renewable_lower", low - power)
                check("renewable_upper", power - high)
                supply += power
            elif kind == "TG":
                power, on = arrays[prefix + "power"], arrays[prefix + "on"]
                check("tg_binary", np.abs(on - np.round(on)))
                check("tg_minimum", float(p["Pgmin"]) * on - power)
                check("tg_maximum", power - float(p["Pgmax"]) * on)
                check("tg_initial_off", np.abs(on[0]))
                check("tg_ramp_up", np.diff(power, axis=0) - float(p["Ug"]))
                check("tg_ramp_down", -np.diff(power, axis=0) - float(p["Dg"]))
                supply += power
                if stage == "rt":
                    costs += float(p["ag"]) * power + float(p["cg"]) * power**2 + float(p["bg"]) * on
            elif kind == "ESS":
                charge, discharge, soc = (arrays[prefix + name] for name in ("charge", "discharge", "soc"))
                charging, discharging = (arrays[prefix + name] for name in ("charging", "discharging"))
                check("ess_binary", np.maximum(np.abs(charging-np.round(charging)), np.abs(discharging-np.round(discharging))))
                check("ess_nonnegative", np.maximum(-charge, -discharge))
                check("ess_charge_power", charge - float(p["p_cmax"]) * asset["power_fraction"] * charging)
                check("ess_discharge_power", discharge - float(p["p_dmax"]) * asset["power_fraction"] * discharging)
                check("ess_exclusivity", charging + discharging - 1)
                check("ess_soc_lower", float(p["socmin"]) - soc)
                check("ess_soc_upper", soc - float(p["socmax"]))
                predicted = 0.5 + np.cumsum(DT * (float(p["yt_sin"]) * charge - discharge / float(p["yt_sout"])), axis=0) / (float(p["Esmax"]) * DT)
                check("ess_energy_balance", np.abs(soc - predicted))
                check("ess_terminal", np.abs(soc[-1] - 0.5))
                supply += discharge - charge
                if stage == "rt":
                    costs += float(p["as"]) * (charge + discharge)
            else:
                power = arrays[prefix + "power"]
                load = p["Load_base"][:, None] if stage == "da" else p["Load_f"]
                check("dr_nonnegative", -power)
                check("dr_maximum", power - 0.85 * load)
                supply += power
                if stage == "rt":
                    costs += float(p["Cutting"]) * power
        supplies[stage] = supply
    bid, delivered, active = arrays["bid"], arrays["delivered"], arrays["delivery_active"]
    check("delivery_binary", np.abs(active-np.round(active)))
    check("day_ahead_balance", np.abs(bid[:, None] - supplies["da"]))
    check("delivered_supply", delivered - supplies["rt"])
    check("delivered_nonnegative", -delivered)
    check("delivery_upper", delivered - 1.3 * bid[:, None])
    check("delivery_active_lower", np.where(active > .5, .8 * bid[:, None] - delivered, 0))
    check("inactive_zero", np.where(active <= .5, np.abs(delivered), 0))
    deviation = delivered - bid[:, None]
    check("absolute_deviation", np.abs(arrays["absolute_deviation"] - np.abs(deviation)))
    revenue_da = float(np.sum(p["Lmpda"] * bid[:, None] * p["P"]))
    revenue_rt = float(np.sum(p["Lmprt"] * deviation * p["P"]))
    penalty = float(np.sum(float(p["pane"]) * np.abs(deviation) * p["P"]))
    operation_cost = float(np.sum(costs * p["P"]))
    return {"maximum_violation": max(violations.values(), default=0.), "constraint_residuals": violations,
            "day_ahead_revenue": revenue_da, "real_time_deviation_revenue": revenue_rt,
            "deviation_penalty": penalty, "operating_cost": operation_cost,
            "reconstructed_objective": revenue_da + revenue_rt - penalty - operation_cost,
            "energy_capacity_mwh": float(p["Esmax"]) * DT, "participation_hours": 24 * DT}
