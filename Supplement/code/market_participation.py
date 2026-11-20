"""Compute a coalition's two-stage electricity-market participation profit using COPT."""
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter

import coptpy as cp
import numpy as np

HERE = Path(__file__).resolve().parent


def solve(params, members, limit, threads, relative_gap):
    p = params
    renewables, turbines, batteries, loads = p["counts"]
    n, scenarios = p["Lmpda"].shape
    if not np.isclose(sum(p["P"]), 1.) or len(set(members)) != len(members):
        raise ValueError("Invalid scenario probabilities or repeated member")
    if not members or min(members) < 1 or max(members) > sum(p["counts"]):
        raise ValueError("Members must be one-based indices of a non-empty coalition")
    env = cp.Envr()
    model = env.createModel("day_ahead_recourse")
    for key, val in {"Logging":0, "Threads":threads, "TimeLimit":limit, "RelGap":relative_gap,
                     "AbsGap":1e-6, "FeasTol":1e-8, "IntTol":1e-8}.items():
        model.setParam(key, val)
    def var(ub=None):
        return model.addVar(lb=0) if ub is None else model.addVar(lb=0, ub=float(ub))

    def binary():
        return model.addVar(vtype=cp.COPT.BINARY)

    bids = [var() for _ in range(n)]
    objective = cp.QuadExpr()
    # w=-1 is the common day-ahead schedule; nonnegative w indexes recourse.
    for w in range(-1, scenarios):
        supply = [cp.LinExpr() for _ in range(n)]
        cost = cp.QuadExpr()
        for member in members:
            k = member - 1
            if k < renewables:
                actual = p["renewable"][k]
                for t in range(n):
                    if w == -1:
                        quantity = model.addVar(lb=float(min(actual[t])), ub=float(max(actual[t])))
                        supply[t] += quantity
                    else:
                        supply[t] += float(actual[t,w])
            elif k < renewables + turbines:
                i = k - renewables
                previous = 0
                for t in range(n):
                    power, on = var(), binary()
                    model.addConstr(power <= float(p["Pgmax"][i]) * on)
                    model.addConstr(power >= float(p["Pgmin"][i]) * on)
                    if t == 0:
                        model.addConstr(on == 0)
                    else:
                        model.addConstr(power-previous <= float(p["Ug"][i]))
                        model.addConstr(previous-power <= float(p["Dg"][i]))
                    supply[t] += power
                    if w >= 0:
                        cost += float(p["ag"][i])*power + float(p["bg"][i])*on + float(p["cg"][i])*power*power
                    previous = power
            elif k < renewables + turbines + batteries:
                i = k - renewables - turbines
                previous = 0.5
                for t in range(n):
                    ch, dis, charging = var(), var(), binary()
                    # One binary suffices: idle is allowed in either mode.
                    model.addConstr(ch <= float(p["p_cmax"][i])*charging)
                    model.addConstr(dis <= float(p["p_dmax"][i])*(1-charging))
                    soc = model.addVar(lb=float(p["socmin"][i]), ub=float(p["socmax"][i]))
                    model.addConstr(soc == previous + (float(p["yt_sin"][i])*ch - dis / float(p["yt_sout"][i])) / float(p["Esmax"][i]))
                    if t == n-1:
                        model.addConstr(soc == 0.5)
                    supply[t] += dis - ch
                    if w >= 0:
                        cost += float(p["as"][i])*(ch+dis)
                    previous = soc
            else:
                i = k - renewables - turbines - batteries
                for t in range(n):
                    available = p["dr_da"][i,t] if w == -1 else p["dr_rt"][i,t,w]
                    cut = var(0.85 * available)
                    supply[t] += cut
                    if w >= 0:
                        cost += float(p["Cutting"][i])*cut
        if w == -1:
            for t in range(n):
                model.addConstr(bids[t] == supply[t])
            continue
        revenue = cp.LinExpr()
        for t in range(n):
            delivery, deviation, active = var(), var(), binary()
            bid = bids[t]
            model.addConstr(delivery <= supply[t])
            model.addConstr(delivery <= 1.3*bid)
            model.addGenConstrIndicator(active, False, delivery == 0)
            model.addGenConstrIndicator(active, True, delivery >= 0.8*bid)
            model.addConstr(deviation >= delivery-bid)
            model.addConstr(deviation >= bid-delivery)
            revenue += (float(p["Lmpda"][t,w])*bid + float(p["Lmprt"][t,w])*(delivery-bid)
                        - float(p["pane"][0])*deviation)
        objective += float(p["P"][w])*(revenue-cost)
    model.setObjective(objective, cp.COPT.MAXIMIZE)
    model.solve()
    return {"status":model.status, "optimal_within_tolerance":model.status == cp.COPT.OPTIMAL,
            "lower":float(model.bestobj) if model.hasmipsol else None,
            "upper":float(model.bestbnd) if abs(model.bestbnd) < 1e25 else None,
            "variables":model.cols, "binary":model.bins, "scenarios":scenarios}

