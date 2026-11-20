"""Reconstruct four-VPP market coefficients from coalition values, retaining revised contracts."""
import argparse
import csv
from functools import lru_cache
from itertools import combinations
from math import comb
import json
from pathlib import Path

import coptpy as cp
import numpy as np
from scipy.io import loadmat, savemat

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "large/data/twenty_case.mat"


def subsets(members):
    for size in range(len(members)+1):
        yield from map(frozenset, combinations(sorted(members),size))


def nucleolus(portfolio, surplus, env):
    """Sequential COPT least-core LPs with universal tightness tested on each optimal face."""
    ids = sorted(portfolio)
    proper = [s for s in subsets(portfolio) if 0 < len(s) < len(ids)]
    coal = np.array([[int(j in s) for j in ids] for s in proper])
    scale = max(1., abs(surplus(portfolio)), max(abs(surplus(s)) for s in proper))
    worth = np.array([surplus(s)/scale for s in proper])
    fixed = [np.ones(len(ids))]
    values = [surplus(portfolio)/scale]
    remaining = list(range(len(proper)))
    while len(fixed) < len(ids):
        m = env.createModel()
        for key,val in {"Logging":0, "Threads":1,"FeasTol":1e-10,"DualTol":1e-10}.items():
            m.setParam(key,val)
        x = [m.addVar(lb=0.) for _ in ids]
        eps = m.addVar(lb=-cp.COPT.INFINITY)
        def allocated(row):
            return cp.quicksum(x[j] for j in np.flatnonzero(row))
        amounts = {k:allocated(coal[k]) for k in remaining}
        for eq,target in zip(fixed,values):
            m.addConstr(allocated(eq)==target)
        for k in remaining:
            m.addConstr(amounts[k]+eps >= float(worth[k]))
        m.setObjective(eps,cp.COPT.MINIMIZE)
        m.solve()
        if m.status != cp.COPT.OPTIMAL:
            raise RuntimeError("Least-core LP failed")
        level = eps.x
        allocation = np.array([v.x for v in x])
        excess = worth-coal @ allocation
        candidates = [k for k in remaining if abs(excess[k]-level) < 1e-7]
        m.addConstr(eps <= level + 1e-10)
        tight = set()
        for k in candidates:
            m.setObjective(amounts[k],cp.COPT.MAXIMIZE)
            m.solve()
            if m.status != cp.COPT.OPTIMAL:
                raise RuntimeError("Nucleolus optimal-face LP failed")
            if m.objval + level - worth[k] <= 1e-7:
                tight.add(k)
                if np.linalg.matrix_rank([*fixed,coal[k]]) > len(fixed):
                    fixed.append(coal[k])
                    values.append(worth[k]-level)
        if not tight:
            raise ArithmeticError("No universally tight proper coalition")
        remaining = [k for k in remaining if k not in tight]
    allocation = np.linalg.solve(np.array(fixed),np.array(values))*scale
    if np.min(allocation) < -1e-5 or abs(sum(allocation)-surplus(portfolio))>1e-5:
        raise ArithmeticError("Allocation fails rationality or efficiency")
    return dict(zip(ids, allocation))


def build(records, game):
    values = {frozenset(map(int,k.split(","))):float(v) for k,v in records.items() if k}
    values[frozenset()] = 0.
    @lru_cache(None)
    def surplus(group):
        return values[group]-sum(values[frozenset([j])] for j in group)
    env = cp.Envr()
    @lru_cache(None)
    def allocate(group, method):
        if method == 3:
            return nucleolus(group,surplus,env)
        if method == 1:
            margins = {j:surplus(group)-surplus(group-{j}) for j in group}
            return {j:v*surplus(group)/sum(margins.values()) for j,v in margins.items()}
        result = {}
        for j in group:
            others = group-{j}
            result[j] = sum((surplus(s|{j})-surplus(s))/(len(group)*comb(len(others),len(s)))
                            for s in subsets(others))
        return result
    f = np.empty((20,4))
    sc = np.empty((4,20))
    iv = np.empty((1,4))
    for v, entry in enumerate(game["cfg"]["vpp"]):
        current = frozenset(map(int,entry["members"]))
        iv[0,v] = surplus(current)
        for j in range(1,21):
            group = current|{j}
            f[j-1,v] = allocate(group,int(entry["allocation_method"]))[j]
            contribution = surplus(group)-surplus(group-{j})
            sc[v,j-1] = -contribution if j in current else contribution
    return dict(FV=f,Sc=sc,IV=iv)


def as_record(path):
    with path.open(encoding="utf-8-sig",newline="") as fp:
        rows = list(csv.DictReader(fp))
    # A resumed run may contain multiple attempts. Retain the strongest feasible
    # incumbent for each maximization problem, even if a later attempt times out.
    result = {}
    for r in rows:
        if r["lower"] not in ("", None):
            k = r["members"]
            result[k] = max(float(r["lower"]), result.get(k, float("-inf")))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", type=Path, help="CSV with members (one-based CSV list) and lower (profit)")
    parser.add_argument("--mat",type=Path,default=HERE / "twenty_end_to_end.mat")
    args = parser.parse_args()
    source = loadmat(FROZEN,simplify_cells=True)
    ref = json.loads((HERE / "inputs/reference.json").read_bytes())["twenty"]
    rebuilt = build(ref,source)
    def differences(actual):
        return {k:np.abs(x - np.asarray(source["Parameter"][k])).max().item() for k,x in actual.items()}
    err = differences(rebuilt)
    if max(err.values()) > 1e-5:
        raise ArithmeticError(f"Allocation differs from manuscript MAT: {err}")
    out = dict(archived_coefficient_max_abs_errors=err, source_table_rows=len(ref))
    if args.dispatch:
        dispatched = as_record(args.dispatch)
        if set(dispatched) != set(ref):
            raise ValueError(f"Expected all {len(ref)} coalition incumbents, received {len(dispatched)}")
        fresh = build(dispatched,source)
        raw = {k:v for k,v in loadmat(FROZEN).items() if not k.startswith("_")}
        for k,v in fresh.items():
            if raw["Parameter"][k][0,0].shape != v.shape:
                raise ValueError(f"Unexpected shape for {k}")
            raw["Parameter"][k][0,0] = v
        savemat(args.mat,raw)
        # A fixed format-v5 header gives identical files on an exact replay.
        with args.mat.open("r+b") as handle:
            handle.write(b"MATLAB 5.0 MAT-file, reconstructed COPT dispatch and allocations".ljust(116))
        saved = loadmat(args.mat,simplify_cells=True)["Parameter"]
        for key in ("CV","Switching"):
            np.testing.assert_array_equal(saved[key],source["Parameter"][key])
        out.update(dispatch_file=args.dispatch.name, mat=args.mat.name, max_differences=differences(fresh),
                   current_surplus=fresh["IV"].ravel().tolist())
    path = HERE/("twenty_rebuild.json" if args.dispatch else "twenty_allocation_check.json")
    path.write_text(json.dumps(out,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()

