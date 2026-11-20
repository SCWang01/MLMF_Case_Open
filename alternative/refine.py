"""Tighten four-VPP numerical lower bounds with MILP relaxations on nested boxes.

Each root starts with historically feasible deviation functions. Published
objective bounds are never loaded into a relaxation. A parent's lower bound
applies to both children, so the minimum over the active cover remains valid.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import coptpy as cp
import numpy as np

from run_twenty import (CASES, HERE, ROOT, ExactGame, add_response_rules, exact_upper,
                        load_game, read, save, solve_box)


def inside(point, box):
    return np.all(np.asarray(point)>=box[:,0]) and np.all(np.asarray(point)<=box[:,1])


def run_region(job):
    case,name,spec,known_point,target,limit,seconds,budget,presolve,previous=job
    game,_=load_game(CASES[case])
    exact=ExactGame(CASES[case])
    env=cp.Envr()
    rules=[list(group) for group in spec["rules"]]
    initial=np.array(spec["bounds"],dtype=float)
    witnesses=[(spec["audit_quote"],exact_upper(exact,spec["audit_quote"]))]
    if inside(known_point,initial):
        witnesses.append((known_point,exact_upper(exact,known_point)))
    best_quote,best=min(witnesses,key=lambda pair:pair[1])
    started=time.monotonic()
    steps=[]
    leaves={}
    previous_count=0

    def assess(key,box,parent_lower):
        nonlocal best,best_quote
        raw=solve_box(game,box,rules,seconds=seconds,threads=1,presolve=presolve,env=env)
        q=raw["candidate"]
        if q is not None:
            ub=exact_upper(exact,q)
            if ub<best:
                best,best_quote=ub,q
            witnesses.append((q,ub))
        check=min((value for point,value in witnesses if inside(point,box)),default=float("inf"))
        if parent_lower>check:
            raise ArithmeticError(f"Inherited bound conflict at {case}/{name}/{key}")
        conflict=raw["conflict"] or raw["lower"]>check
        bound=max(parent_lower,0. if conflict else raw["lower"])
        scores=np.array(raw.get("split_scores",[0.]*4))
        widths=box[:,1]-box[:,0]
        axis=int(np.argmax(scores)) if scores.max()>1e-12 else int(np.argmax(widths))
        # Protect against repeatedly branching a nearly fixed coordinate.
        if widths[axis]<1e-8*widths.max():
            axis=int(np.argmax(widths))
        steps.append(dict(path=key,bounds=box.tolist(),raw_bound=raw["raw_bound"],lower=bound,
                          status=raw["status"],conflict=conflict,candidate=q,axis=axis,
                          rules=sum(map(len,rules)),seconds=raw["seconds"],presolve=presolve,lower_of_cover=None))
        return dict(box=box,lower=bound,axis=axis,quote=q)

    if previous is None:
        leaves[""]=assess("",initial,0.)
        steps[-1]["lower_of_cover"]=leaves[""]["lower"]
    else:
        check_cover(previous)
        steps=list(previous["steps"])
        previous_count=len(steps)
        nodes={s["path"]:s for s in steps}
        if previous["upper"]<best:
            best,best_quote=previous["upper"],previous["quote"]
        for s in steps:
            point=s["candidate"]
            if point is not None:
                witnesses.append((point,exact_upper(exact,point)))
            if s["path"].endswith("L"):
                previous_quote=nodes[s["path"][:-1]]["candidate"]
                if previous_quote is not None:
                    add_response_rules(game,rules,previous_quote)
        for key,lb in previous["leaves"].items():
            s=nodes[key]
            leaves[key]=dict(box=np.asarray(s["bounds"]),lower=lb,axis=s["axis"],quote=s["candidate"])
    while len(steps)-previous_count+2<=limit and time.monotonic()-started<budget:
        key,node=min(leaves.items(),key=lambda p:(p[1]["lower"],len(p[0]),p[0]))
        if node["lower"]>=target:
            break
        q=node["quote"]
        if q is not None:
            add_response_rules(game,rules,q)
        axis=node["axis"]
        box=node["box"]
        mid=(box[axis,0]+box[axis,1])/2.
        if not box[axis,0]<mid<box[axis,1]:
            break
        left,right=box.copy(),box.copy()
        left[axis,1]=mid
        right[axis,0]=mid
        del leaves[key]
        for suffix,child in (("L",left),("R",right)):
            leaves[key+suffix]=assess(key+suffix,child,node["lower"])
        steps[-1]["lower_of_cover"]=min(r["lower"] for r in leaves.values())
    lower=min(r["lower"] for r in leaves.values())
    if lower>best:
        raise ArithmeticError("Root bound conflict")
    return dict(case=case,name=name,lower=lower,upper=best,quote=best_quote,
                root_lower=steps[0]["lower"],target=target,
                seconds=time.monotonic()-started+(previous["seconds"] if previous else 0.),steps=steps,leaves={k:r["lower"] for k,r in leaves.items()})


def check_cover(row):
    root=row["steps"][0]["bounds"]
    records={s["path"]:s for s in row["steps"]}
    if len(records)!=len(row["steps"]):
        raise ValueError("Duplicate node")
    pending=[("",np.asarray(root),0.)]
    seen=set()
    while pending:
        path,box,lower=pending.pop()
        node=records[path]
        if not np.array_equal(node["bounds"],box) or node["lower"]<lower:
            raise ValueError("Broken bisection cover")
        seen.add(path)
        if path not in row["leaves"]:
            dim=node["axis"]
            left,right=box.copy(),box.copy()
            mid=(left[dim,0]+right[dim,1])/2.
            left[dim,1]=mid
            right[dim,0]=mid
            pending.extend([(path+"L",left,node["lower"]),(path+"R",right,node["lower"])])
        elif row["leaves"][path]!=node["lower"]:
            raise ValueError("Wrong terminal bound")
    if seen!=set(records) or min(row["leaves"].values())!=row["lower"]:
        raise ValueError("Incomplete cover")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=["frozen"],default="frozen")
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--nodes",type=int,default=81,help="maximum number of MILP masters per original box")
    p.add_argument("--seconds",type=float,default=3.,help="COPT limit per MILP master")
    p.add_argument("--budget",type=float,default=120.,help="wall time per original box, checked between splits")
    p.add_argument("--presolve",type=int,choices=[-1,0],default=-1)
    p.add_argument("--continue-from",type=Path,help="continue complete box records from a matching earlier run")
    p.add_argument("--output",type=Path,default=HERE/"refinement")
    args=p.parse_args()
    if min(args.nodes,args.seconds,args.budget,args.workers)<=0:
        p.error("positive budgets required")
    spec=read(ROOT/"large/data/refinement_rules.json")
    cases=list(CASES) if args.case=="both" else [args.case]
    results={c:{} for c in cases}
    jobs=[]
    for name,leaf in spec["boxes"].items():
        for case in cases:
            point=read(ROOT/("verification/candidate_"+case+".json"))["quote"]
            target=98.4051 if case=="frozen" else 97.2451
            previous=None
            if args.continue_from:
                meta=read(args.continue_from/(case+".json"))
                if meta["sha256"]!=hashlib.sha256(CASES[case].read_bytes()).hexdigest() or meta["engine_sha256"]!=hashlib.sha256((HERE/"milp.py").read_bytes()).hexdigest():
                    raise ValueError("Cached case or MILP formulation differs")
                cached=args.continue_from/case/(name+".json")
                if cached.exists():
                    previous=read(cached)
                    check_cover(previous)
                    if previous["lower"]!=meta["regions"][name]["lower"]:
                        raise ValueError("Cached record differs from its summary")
                    if previous["lower"]>=target:
                        results[case][name]=previous
                        continue
            jobs.append((case,name,leaf,point,target,args.nodes,args.seconds,args.budget,args.presolve,previous))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_region,j) for j in jobs]):
            r=future.result()
            check_cover(r)
            save(args.output/r["case"]/(r["name"]+".json"),r)
            results[r["case"]][r["name"]]=r
            print(r["case"],r["name"],round(r["lower"],6),len(r["steps"]),flush=True)
    for case,items in results.items():
        if set(items)!=set(spec["boxes"]):
            raise ValueError("Incomplete root cover")
        for name,row in items.items():
            save(args.output/case/(name+".json"),row)
        best=min(items.values(),key=lambda r:r["upper"])
        out=dict(case=case,sha256=hashlib.sha256(CASES[case].read_bytes()).hexdigest(),
                 engine_sha256=hashlib.sha256((HERE/"milp.py").read_bytes()).hexdigest(),
                 target=best["target"],lower=min(r["lower"] for r in items.values()),
                 upper=best["upper"],quote=best["quote"],root_lower=min(r["root_lower"] for r in items.values()),
                 roots_excluding=sum(r["lower"]>.001001 for r in items.values()),
                 initial_roots_excluding=sum(r["root_lower"]>.001001 for r in items.values()),
                 masters=sum(len(r["steps"]) for r in items.values()),active_leaves=sum(len(r["leaves"]) for r in items.values()),
                 conflicts=sum(s["conflict"] for r in items.values() for s in r["steps"]),
                 settings=dict(seconds=args.seconds,budget=args.budget,nodes=args.nodes,workers=args.workers,presolve=args.presolve,
                               continued_from=args.continue_from.name if args.continue_from else None),
                 regions={n:{k:r[k] for k in ("lower","upper","root_lower","seconds")} for n,r in items.items()})
        if out["lower"]>out["upper"]:
            raise ArithmeticError("Conflicting full-domain bounds")
        save(args.output/(case+".json"), out)
        print("RESULT",case,out["lower"],out["upper"],out["masters"],flush=True)

if __name__=="__main__":
    main()
