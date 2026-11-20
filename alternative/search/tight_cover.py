"""Dyadic subdivision with exact replay of the chosen feasible deviations.

Split the widest dimension of each unresolved candidate box. A terminal
box stores the feasible deviation rules examined by the integer checker in
verification/exact_four.py. Replaying the certificate does not import COPT.
"""
import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/"verification"))
from exact_four import K, boxes, read_game, scaled, check_leaf


def split_region(region,axis):
    lo,hi=region[axis]
    if (lo+hi)%2 or lo>=hi:
        raise ValueError("Insufficient dyadic resolution")
    mid=(lo+hi)//2
    return (region[:axis]+[(lo,mid)]+region[axis+1:],
            region[:axis]+[(mid,hi)]+region[axis+1:])


def generate(src,target,limit):
    from weighted import compute
    from advanced_four import load_game
    game,_=load_game(src)
    params=(game.f,game.c,game.contribution,game.compensation,game.base,game.epsilon)
    exact=read_game(src)
    t=float(target)+1e-6
    rules=np.empty(4,np.int64)
    quotes=np.empty(4)
    mix=np.ones(4,np.int64)
    count=0
    def visit(region):
        nonlocal count
        count+=1
        if count>limit:
            raise RuntimeError(f"Node budget exhausted ({limit})")
        approx=np.array([[a/K,b/K] for a,b in region])
        bound,mask=compute(*params,approx,t,rules,quotes,mix)
        if bound>t:
            witness={str(v):(["capture",int(rules[v])] if rules[v]>=0 else
                              ["fixed",str(scaled(quotes[v]))]) for v in range(4) if mask>>v&1}
            leaf = {"rules":witness,"multipliers":{v:int(mix[int(v)]) for v in witness}}
            if checked_bound(exact,region,leaf)>target:
                return leaf
        axis=int(np.argmax(approx[:,1]-approx[:,0]))
        left,right=split_region(region,axis)
        return [int(axis),visit(left),visit(right)]
    trees={}
    for key,root in boxes().items():
        trees[key]=visit(root)
        print(key,count,flush=True)
    return dict(input_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                target=str(target),trees=trees)


def checked_bound(game,region,leaf):
    if "multipliers" not in leaf:
        return check_leaf(game,region,leaf)
    rules=leaf["rules"]
    m=leaf["multipliers"]
    if not rules or set(m)!=set(rules) or not set(m)<=set("0123") or any(type(x)!=int or x<=0 for x in m.values()):
        raise ValueError("Invalid deviation multipliers")
    if all(v==1 for v in m.values()):
        return check_leaf(game,region,rules)
    f,c,s,fee,base,eps=game
    weights=[m.get(str(v),1) for v in range(4)]
    scaled_game=(f,c,[[e*w for e in row] for row,w in zip(s,weights)],
                 [[e*w for e in row] for row,w in zip(fee,weights)],[e*w for e,w in zip(base,weights)],eps)
    return len(rules)*check_leaf(scaled_game,region,rules)/sum(m.values())


def verify(src,record,target):
    if record["input_sha256"]!=hashlib.sha256(src.read_bytes()).hexdigest():
        raise ValueError("Input hash mismatch")
    if Fraction(record["target"])!=target:
        raise ValueError("Certificate target mismatch")
    roots=boxes()
    if roots.keys()!=record["trees"].keys():
        raise ValueError("Incomplete root cover")
    game=read_game(src)
    counts=0
    least={}
    for key,region in roots.items():
        stack=[(region,record["trees"][key])]
        bounds=[]
        while stack:
            box,tree=stack.pop()
            if isinstance(tree,dict):
                if not tree or ("multipliers" not in tree and not set(tree)<=set("0123")):
                    raise ValueError("Invalid witness player index")
                val=checked_bound(game,box,tree)
                if val<=target:
                    raise ArithmeticError(f"Lower bound contradicts target in {key}")
                bounds.append(val)
                counts+=1
            else:
                axis,left,right=tree
                if type(axis)!=int or not 0<=axis<4:
                    raise ValueError("Invalid partition coordinate")
                b0,b1=split_region(box,axis)
                stack.extend(((b0,left),(b1,right)))
        least[key]=min(bounds)
    lower=min(least.values())
    return dict(input_sha256=record["input_sha256"],target=str(target),leaves=counts,
                box_lowers={k:float(v) for k,v in least.items()},
                global_lower=float(lower),exact_lower=str(lower))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",required=True,choices=("frozen","dispatch"))
    parser.add_argument("--target",help="default uses the recorded four-decimal certificate threshold")
    parser.add_argument("--generate",action="store_true")
    parser.add_argument("--limit",type=int,default=150000)
    args=parser.parse_args()
    if args.target is None:
        args.target = "98.40621" if args.case=="frozen" else "97.24961"
    target=Fraction(args.target)
    src=ROOT/("large/data/twenty_case.mat" if args.case=="frozen" else "physical/twenty_end_to_end.mat")
    artifact=HERE/f"cover_{args.case}_{args.target}.json.gz"
    started=time.monotonic()
    if args.generate:
        record=generate(src,target,args.limit)
        tmp=artifact.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(json.dumps(record,separators=(",",":")).encode(),mtime=0))
        tmp.replace(artifact)
    payload=artifact.read_bytes()
    summary=verify(src,json.loads(gzip.decompress(payload)),target)
    summary["certificate_sha256"]=hashlib.sha256(payload).hexdigest()
    print(args.case,summary["global_lower"],summary["leaves"],time.monotonic()-started,flush=True)
    (HERE/f"bound_{args.case}_{args.target}.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()

