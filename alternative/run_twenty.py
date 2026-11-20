"""Run an endpoint-initialized MILP version of the manuscript's two-stage CCG."""
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys
import time

import coptpy as cp
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/"large"/"core"))
from case20_game import load_game
from minimax_policies import best_response_policy
from solve_minimax_4vpp import capture_action, evaluate_candidate
from audit_minimax_4vpp_candidate import ExactGame
from fractions import Fraction
from milp import solve_box

ENGINE_SHA256 = hashlib.sha256((HERE/"milp.py").read_bytes()).hexdigest()
THRESHOLD = .001001
CASES = {"frozen": ROOT/"large/data/twenty_case.mat"}


def read(path):
    return json.loads(path.read_bytes())


def save(path, payload):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload,allow_nan=False,indent=2),encoding="utf-8")
    tmp.replace(path)


def rule_for(game,v,point,action):
    if 0.<action<1.:
        for offset in (-1,1):
            for d in range(game.d):
                if abs(capture_action(game,v,d,point,offset)-action)<1e-10:
                    return dict(kind="capture" if offset==-1 else "departure",der=d,offset=offset)
    return list(best_response_policy(game,v,point,action))


def add_response_rules(game,rules,point):
    candidate=evaluate_candidate(game,point)
    count=0
    for v,action in enumerate(candidate["best_response_actions"]):
        rule=rule_for(game,v,np.array(point),action)
        if rule not in rules[v]:
            rules[v].append(rule)
            count+=1
    return count


def endpoint_rules(game):
    return [[[float(a),*([0.]*game.v)] for a in (0,1)] for v in range(game.v)]


def exact_upper(game,quote):
    q=tuple(map(Fraction,quote))
    shares=game.shares(q)
    gains=[game.best_response(v,q)[1]-game.payoff(v,q,shares) for v in range(len(q))]
    return float(max(gains))


def solve_case(job):
    case,name,box,audit_point,settings=job
    source=CASES[case]
    game,_=load_game(source)
    exact=ExactGame(source)
    box=np.asarray(box,dtype=float)
    if np.any(audit_point<box[:,0]) or np.any(audit_point>box[:,1]):
        raise ValueError(f"Audit profile outside {name}")
    audit=exact_upper(exact,audit_point)
    env=cp.Envr()
    rules=endpoint_rules(game)
    lower,best,quote,stop=0.,None,None,"round budget"
    history=[]
    started=time.monotonic()
    for step in range(1,settings["rounds"]+1):
        if time.monotonic()-started>settings["total_seconds"]:
            stop="time budget"
            break
        remaining=max(.01,settings["total_seconds"]-(time.monotonic()-started))
        raw=solve_box(game,box,rules,seconds=min(settings["round_seconds"],remaining),
                      threads=1,presolve=settings["presolve"],env=env)
        candidate=raw["candidate"]
        if candidate is not None:
            upper=exact_upper(exact,candidate)
            if best is None or upper<best:
                best,quote=upper,candidate
        audit_limit=audit if best is None else min(best,audit)
        conflict=raw["conflict"] or raw["lower"]>audit_limit
        previous_conflict=lower>audit_limit
        if previous_conflict:
            lower=0.
        if not conflict:
            lower=max(lower,raw["lower"])
        row=dict(round=step,lower=lower,upper=best,candidate=candidate,status=raw["status"],
                 raw_bound=raw["raw_bound"],conflict=conflict or previous_conflict,rules=sum(map(len,rules)),
                 binaries=raw["binary_count"],rows=raw["rows"],seconds=raw["seconds"])
        history.append(row)
        if conflict or previous_conflict:
            stop="bound conflict"
            break
        if lower>THRESHOLD:
            stop="excluded"
            break
        if best is not None and best<=.000999:
            stop="within tolerance"
            break
        if candidate is None:
            stop="no master candidate"
            break
        if add_response_rules(game,rules,candidate)==0:
            stop="no new rule"
            break
    return dict(case=case,box=name,engine_sha256=ENGINE_SHA256,bounds=box.tolist(),settings=settings,stop=stop,lower=lower,
                upper=best,quote=quote,history=history,seconds=time.monotonic()-started,
                audit_upper=audit,rules=rules)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=[*CASES,"both"],default="both")
    p.add_argument("--rounds",type=int,default=6)
    p.add_argument("--seconds",type=float,default=3.,help="local MILP limit per master")
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--presolve",type=int,choices=[-1,0,1],default=-1)
    p.add_argument("--output",type=Path,default=HERE/"decision")
    args=p.parse_args()
    if min(args.rounds,args.workers,args.seconds)<=0:
        p.error("positive round, time and worker budgets required")
    boxes=read(ROOT/"large/data/twenty_boxes.json")
    archived=read(ROOT/"large/data/refinement_rules.json")
    jobs=[]
    for case in (CASES if args.case=="both" else [args.case]):
        for name,bounds in [("full",[[0.,1.]]*4),*boxes.items()]:
            full=name=="full"
            quote=archived["quote"] if full else archived["boxes"][name]["audit_quote"]
            settings=dict(rounds=args.rounds,round_seconds=3*args.seconds if full else args.seconds,
                          total_seconds=120 if full else 60,presolve=args.presolve)
            jobs.append((case,name,bounds,np.array(quote),settings))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(solve_case,job) for job in jobs]
        results={case:{} for case in (CASES if args.case=="both" else [args.case])}
        for future in as_completed(futures):
            item=future.result()
            case,name=item["case"],item["box"]
            results[case][name]=item
            print(case,name,item["stop"],round(item["lower"],5),flush=True)
    for case,items in results.items():
        local=[v for k,v in items.items() if k!="full"]
        if set(v["box"] for v in local)!=set(boxes):
            raise ValueError("Incomplete cover")
        if any(v["engine_sha256"]!=ENGINE_SHA256 for v in items.values()):
            raise ValueError("Mixed MILP engine revisions")
        lower=max(items["full"]["lower"],min(v["lower"] for v in local))
        best=min((v for v in items.values() if v["upper"] is not None),key=lambda v:v["upper"],default=None)
        if best and lower>best["upper"]:
            raise ValueError("Domain bound conflicts with a feasible quote")
        out=dict(case=case,engine_sha256=ENGINE_SHA256,sha256=hashlib.sha256(CASES[case].read_bytes()).hexdigest(),threshold=THRESHOLD,
                 excluded=sum(v["stop"]=="excluded" for v in local),lower=lower,
                 upper=best["upper"] if best else None,best_quote=best["quote"] if best else None,
                 items=items)
        save(args.output/(case+".json"),out)
        print("RESULT",case,out["excluded"],lower,out["upper"],flush=True)


if __name__=="__main__":
    main()
