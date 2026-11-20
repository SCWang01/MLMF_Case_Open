"""Verify a complete four-VPP exclusion with exact, outward-rounded integer bounds.

The integer scale is common to inputs, quotes and shares. Every univariate
product defining an offer is rounded outwards. Division in a simplex share is
rounded down for its lower bound, up for its upper bound. All payoff products
and comparisons are then computed exactly with Python integers.
"""
from argparse import ArgumentParser
from fractions import Fraction
from itertools import combinations
from pathlib import Path
import gzip
import hashlib
import json
import time

import numpy as np
from scipy.io import loadmat

HERE=Path(__file__).resolve().parent
K=1<<112
TARGET=Fraction(1001,1000000)


def scaled(number):
    fraction=Fraction(float(number))*K
    if fraction.denominator!=1:
        raise ValueError("Input exceeds the common dyadic precision")
    return fraction.numerator


def ceil_div(num,den):
    return -((-num)//den)


def clamp(q):
    return min(K,max(0,q))


def scaled_array(values):
    if isinstance(values,list):
        return [scaled_array(a) for a in values]
    return scaled(values)


def read_game(source):
    raw=loadmat(source,simplify_cells=True)
    p=raw["Parameter"]
    f,c,sc,fee,iv=(scaled_array(p[k].tolist()) for k in ("FV","CV","Sc","Switching","IV"))
    owners=[[int(j)-1 for j in v["members"]] for v in raw["cfg"]["vpp"]]
    s=[[-a if d in owners[v] else a for d,a in enumerate(row)] for v,row in enumerate(sc)]
    base=[iv[v]+sum(sc[v][d] for d in group) for v,group in enumerate(owners)]
    return f,c,s,fee,base,scaled(raw["epis"])


def share(diff_ordered,own_offer,eps,upper):
    # Each rival count k gives an affine upper envelope piece of our
    # projected share. There are at most three rivals.
    num=eps+own_offer-diff_ordered[0]
    if num<=0:
        return 0
    if num>=2*eps:
        return K
    total=0
    best=K
    for k,q_rival in enumerate(diff_ordered,1):
        total+=q_rival
        num=(eps+k*own_offer-total)*K
        den=(k+1)*eps
        best=min(best,ceil_div(num,den) if upper else num//den)
    return clamp(best)


def split(box):
    axis=max(range(4),key=lambda i:box[i][1]-box[i][0])
    a,b=box[axis]
    if (a+b)%2 or a>=b:
        raise ValueError("Exhausted dyadic resolution")
    mid=(a+b)//2
    return [box[:axis]+[(a,mid)]+box[axis+1:],box[:axis]+[(mid,b)]+box[axis+1:]]


def check_leaf(game,box,witness):
    f,c,s,fee,base,eps=game
    nd=len(f)
    lower_offer=[[((K-box[v][1])*f[d][v])//K-c[d][v] for v in range(4)] for d in range(nd)]
    upper_offer=[[ceil_div((K-box[v][0])*f[d][v],K)-c[d][v] for v in range(4)] for d in range(nd)]
    # Nonparticipants need share intervals too, since every DER allocates one unit total.
    ylo=[[0]*4 for _ in f]
    yhi=[[K]*4 for _ in f]
    weights=[[0]*4 for _ in f]
    margin=0
    single_bound=None
    for v in range(4):
        others=[j for j in range(4) if j!=v]
        in_group=str(v) in witness
        if in_group:
            rule=witness[str(v)]
            if rule[0]=="fixed":
                zlo=zhi=int(rule[1])
                if not 0<=zlo<=K:
                    raise ValueError("Infeasible constant deviation")
                pivot=-1
            else:
                if rule[0]!="capture" or not 0<=rule[1]<nd:
                    raise ValueError("Unknown deviation rule")
                pivot=rule[1]
                own=f[pivot][v]-c[pivot][v]-eps
                rawlo=(own-max(upper_offer[pivot][j] for j in others))*K//f[pivot][v]
                zlo=clamp(rawlo)
                zhi=clamp(ceil_div((own-max(lower_offer[pivot][j] for j in others))*K,f[pivot][v]))
                # If the unclipped quote might be negative, evaluate this DER
                # with the ordinary share enclosure as well.
                if rawlo<0:
                    pivot=-1
            dv0,dv1=zlo*base[v]*K,zhi*base[v]*K
            qlo,qhi=box[v]
            slope_lo=slope_hi=base[v]*K
            cur0,cur1=qlo*base[v]*K,qhi*base[v]*K
        for d in range(nd):
            highest=sorted((upper_offer[d][j] for j in others),reverse=True)
            lowest=sorted((lower_offer[d][j] for j in others),reverse=True)
            ylo[d][v]=a=share(highest,lower_offer[d][v],eps,False)
            yhi[d][v]=b=share(lowest,upper_offer[d][v],eps,True)
            if in_group:
                sc=s[v][d]
                fees=fee[v][d]*K
                slope_lo+=min(sc*a,sc*b)
                slope_hi+=max(sc*a,sc*b)
                w0,w1=qlo*sc-fees,qhi*sc-fees
                cur0+=max(w0*a,w0*b)
                cur1+=max(w1*a,w1*b)
                if d==pivot:
                    a=b=K
                else:
                    a=share(highest,((K-zhi)*f[d][v])//K-c[d][v],eps,False)
                    b=share(lowest,ceil_div((K-zlo)*f[d][v],K)-c[d][v],eps,True)
                w0,w1=zlo*sc-fees,zhi*sc-fees
                dv0+=min(w0*a,w0*b)
                dv1+=min(w1*a,w1*b)
        if in_group:
            dev=min(dv0,dv1)
            margin+=dev
            # Optimize each affine-in-own-quote current payoff at a box
            # endpoint when its slope has one sign throughout; otherwise
            # enclose the excursion around its dyadic midpoint.
            if slope_lo>=0:
                q=qhi
            elif slope_hi<=0:
                q=qlo
            else:
                q=(qlo+qhi)//2
                margin-=max(-slope_lo,slope_hi)*max(q-qlo,qhi-q)
            margin-=base[v]*q*K
            for d in range(nd):
                weights[d][v]=q*s[v][d]-fee[v][d]*K
            if len(witness)==1:
                single_bound=dev-max(cur0,cur1)
    for d in range(nd):
        mass=K-sum(ylo[d])
        if mass<0 or sum(yhi[d])<K:
            raise ArithmeticError("Inconsistent simplex enclosure")
        margin-=sum(w*a for w,a in zip(weights[d],ylo[d]))
        for j in sorted(range(4),key=lambda j:-weights[d][j]):
            use=min(mass,yhi[d][j]-ylo[d][j])
            margin-=weights[d][j]*use
            mass-=use
            if not mass:
                break
        if mass:
            raise ArithmeticError("Simplex mass left over")
    if single_bound is not None:
        margin=max(margin,single_bound)
    return Fraction(margin,len(witness)*K**3)


def boxes():
    raw=json.loads((HERE.parent/"large/data/twenty_boxes.json").read_bytes())
    result={key:[tuple(map(scaled,r)) for r in v] for key,v in raw.items()}
    volume=sum(np.prod([b-a for a,b in box],dtype=object) for box in result.values())
    if volume!=K**4 or any(not 0<=a<b<=K for box in result.values() for a,b in box):
        raise ValueError("Partition does not have unit volume")
    for a,b in combinations(result.values(),2):
        if all(min(a[i][1],b[i][1])>max(a[i][0],b[i][0]) for i in range(4)):
            raise ValueError("Positive-volume overlap")
    return result


def verify(source,regions,record):
    if record["sha256"]!=hashlib.sha256(source.read_bytes()).hexdigest() or set(regions)!=set(record["leaves"]):
        raise ValueError("Input or box keys differ from certificate")
    exact=read_game(source)
    minima={}
    for key,root in regions.items():
        stack=[("",root)]
        leaves=record["leaves"][key]
        found=set()
        minimum=None
        while stack:
            path,node=stack.pop()
            if path in leaves:
                bound=check_leaf(exact,node,leaves[path])
                if bound<=TARGET:
                    raise ArithmeticError(f"Unverified leaf {key}/{path}: {float(bound)}")
                minimum=bound if minimum is None else min(minimum,bound)
                found.add(path)
            else:
                if len(path)>100:
                    raise ValueError(f"Missing certificate leaf below {key}/{path}")
                a,b=split(node)
                stack.extend([(path+"0",a),(path+"1",b)])
        if found!=set(leaves):
            raise ValueError("Overlapping certificate leaf")
        minima[key]=minimum
    return minima


def generate(source,regions):
    from advanced_four import compute,load_game
    g,_=load_game(source)
    exact=read_game(source)
    all_leaves={}
    for key,root in regions.items():
        leaves={}
        all_leaves[key]=leaves
        stack=[("",root)]
        while stack:
            path,node=stack.pop()
            numeric=np.array([[a/K,b/K] for a,b in node])
            policy=np.full(4,-1,np.int64)
            quote=np.zeros(4)
            bound,mask=compute(g.f,g.c,g.contribution,g.compensation,g.base,g.epsilon,
                               numeric,float(TARGET)+1e-6,policy,quote)
            if bound-1e-6>float(TARGET):
                witness={str(v):(["capture",int(policy[v])] if policy[v]>=0 else ["fixed",str(scaled(quote[v]))])
                         for v in range(4) if mask>>v&1}
                if check_leaf(exact,node,witness)>TARGET:
                    leaves[path]=witness
                    continue
            a,b=split(node)
            stack.extend([(path+"0",a),(path+"1",b)])
    return dict(sha256=hashlib.sha256(source.read_bytes()).hexdigest(),leaves=all_leaves)


