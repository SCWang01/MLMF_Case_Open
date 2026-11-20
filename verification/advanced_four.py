"""Independent bisection bound using capture deviations and shared simplex constraints.

For every quote box and each player, bound from below the payoff after a
feasible unilateral deviation. A capture deviation guarantees unit share of
its target whenever its unclipped quote stays nonnegative. Current allocations
are coupled by one simplex per DER. Sum the regret inequalities for subsets of
players and maximize their combined current payoff over those joint simplices.
A positive average gain implies at least one player gains that much. All
computations use binary64 and a 1e-6 subtraction from each leaf bound.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from numba import njit

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"large/core"))
from case20_game import load_game


@njit
def sort3(a,b,c):
    if a>b:
        a,b=b,a
    if b>c:
        b,c=c,b
    if a>b:
        a,b=b,a
    return a,b,c


@njit
def share(q,own_slope,table):
    # table lists the three highest rival offers in descending order, folded
    # into normalized ratios at which each projection piece crosses zero.
    t=table[0]-q*own_slope
    if t<=0.:
        return 0.
    if t>=1.:
        return 1.
    return max(0.,min(t,table[1]-q*own_slope*4./3.,table[2]-q*own_slope*1.5))


@njit
def compute(f,c,s,fee,base,eps,box,threshold,policy_id,policy_quote):
    nd,nv=f.shape
    miny=np.empty((nd,4))
    maxy=np.empty((nd,4))
    best=np.empty(4)
    up=np.empty(4)
    slopes=np.empty((nd,4))
    extra=np.empty(4)
    current_weight=np.empty((nd,4))
    best_bound=-1e30
    for v in range(4):
        lo,hi=box[v]
        plane=np.empty((2,nd,3))
        candidates=np.empty(nd*4+2)
        candidates[0],candidates[1]=0.,1.
        n=2
        captures=np.empty((nd,2))
        for d in range(nd):
            fv=f[d,v]
            own=fv-c[d,v]
            slope=fv/(2*eps)
            slopes[d,v]=slope
            for edge in range(2):
                a=(v+1)%4
                b=(v+2)%4
                h=(v+3)%4
                t1,t2,t3=sort3((1-box[a,edge])*f[d,a]-c[d,a],
                                (1-box[b,edge])*f[d,b]-c[d,b],
                                (1-box[h,edge])*f[d,h]-c[d,h])
                plane[edge,d,0]=(eps+own-t3)/(2*eps)
                plane[edge,d,1]=(eps+2*own-t3-t2)/(3*eps)
                plane[edge,d,2]=(eps+3*own-t3-t2-t1)/(4*eps)
                cap=(own-eps-t3)/fv
                captures[d,edge]=cap
                only=min(plane[edge,d,0]/slope,plane[edge,d,1]/(4*slope/3),plane[edge,d,2]/(1.5*slope))
                for q in (cap,only):
                    if 0.<q<1.:
                        candidates[n]=q
                        n+=1
            miny[d,v]=share(hi,slope,plane[0,d])
            maxy[d,v]=share(lo,slope,plane[1,d])
        best_dev=-1e30
        for j in range(n+nd):
            pivot=j-n
            if pivot<0:
                qlo=candidates[j]
                qhi=qlo
            else:
                rawlo,rawhi=captures[pivot]
                if rawlo < 0. or rawlo >= 1.:
                    continue
                qlo=min(1.,rawlo)
                qhi=min(1.,rawhi)
            val0,val1=qlo*base[v],qhi*base[v]
            for d in range(nd):
                if d==pivot:
                    ymin,ymax=1.,1.
                else:
                    ymin=share(qhi,slopes[d,v],plane[0,d])
                    ymax=share(qlo,slopes[d,v],plane[1,d])
                a=qlo*s[v,d]-fee[v,d]
                b=qhi*s[v,d]-fee[v,d]
                val0+=min(a*ymin,a*ymax)
                val1+=min(b*ymin,b*ymax)
            improvement=min(val0,val1)
            if improvement>best_dev:
                best_dev=improvement
                policy_id[v]=pivot if pivot>=0 else -1
                policy_quote[v]=qlo
        best[v]=best_dev # Fee constants cancel between current and deviation.
        a0,a1=lo*base[v],hi*base[v]
        sl,sh=base[v],base[v]
        for d in range(nd):
            y0,y1=miny[d,v],maxy[d,v]
            w0,w1=lo*s[v,d]-fee[v,d],hi*s[v,d]-fee[v,d]
            a0+=max(w0*y0,w0*y1)
            a1+=max(w1*y0,w1*y1)
            sl+=min(s[v,d]*y0,s[v,d]*y1)
            sh+=max(s[v,d]*y0,s[v,d]*y1)
        up[v]=max(a0,a1)
        best_bound=max(best_bound,best_dev-up[v])
        if best_bound>threshold:
            return best_bound,1<<v
        if sl>=0.:
            q=hi
            penalty=0.
        elif sh<=0.:
            q=lo
            penalty=0.
        else:
            q=0.5*(lo+hi)
            penalty=max(-sl,sh)*0.5*(hi-lo)
        extra[v]=base[v]*q+penalty
        for d in range(nd):
            current_weight[d,v]=q*s[v,d]-fee[v,d]
    selected=1<<np.argmax(best-up)
    for mask in (3,5,9,6,10,12,7,11,13,14,15):
        count=0
        lb=0.
        for v in range(4):
            if (mask>>v)&1:
                count+=1
                lb+=best[v]-extra[v]
        for d in range(nd):
            w=np.empty(4)
            mass=1.
            for v in range(4):
                w[v]=current_weight[d,v] if (mask>>v)&1 else 0.
                lb-=w[v]*miny[d,v]
                mass-=miny[d,v]
            # Exact greedy LP on [miny,maxy] with a simplex equality.
            for rank in range(4):
                v=np.argmax(w)
                assigned=max(0.,min(mass,maxy[d,v]-miny[d,v]))
                lb-=w[v]*assigned
                mass-=assigned
                w[v]=-1e30
                if mass<=1e-14:
                    break
        if lb/count>best_bound:
            best_bound=lb/count
            selected=mask
        if best_bound>threshold:
            break
    return best_bound,selected


@njit(nogil=True)
def cover(f,c,s,fee,base,eps,box,target,limit):
    stack=np.empty((150,4,2))
    stack[0]=box
    size=1
    visited=0
    minimum=1e30
    while size>0 and visited<limit:
        size-=1
        node=stack[size].copy()
        bound,_=compute(f,c,s,fee,base,eps,node,target+1e-6,np.empty(4,np.int64),np.empty(4))
        bound-=1e-6
        visited+=1
        if bound>target:
            minimum=min(minimum,bound)
            continue
        widths=node[:,1]-node[:,0]
        dim=np.argmax(widths)
        x=np.mean(node[dim])
        if size+1>=len(stack) or widths[dim]<1e-12:
            return 0.,visited
        stack[size]=node
        stack[size,dim,1]=x
        stack[size+1]=node
        stack[size+1,dim,0]=x
        size+=2
    return (minimum if size==0 else 0.),visited


