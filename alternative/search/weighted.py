"""Weighted pair extension of verification/advanced_four.py for four-operator cells."""
from pathlib import Path
import sys
import numpy as np
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"verification"))
from advanced_four import share, sort3

@njit
def compute(f,c,s,fee,base,eps,box,threshold,policy_id,policy_quote,mix):
    mix[:]=1
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
    if best_bound >= threshold:
        return best_bound,selected
    # At a shared DER, a weighted pair of operators can cancel the effect
    # of a moving allocation. Try the ratio at each crossover of the two
    # marginal-payoff lines. Integer weights are recorded for exact replay.
    denom=65536
    for i in range(4):
        for j in range(i+1,4):
            for pivot in range(nd):
                if maxy[pivot,i]-miny[pivot,i]<1e-10 or maxy[pivot,j]-miny[pivot,j]<1e-10:
                    continue
                a,b=current_weight[pivot,i],current_weight[pivot,j]
                if a*b<=0.:
                    continue
                wi=int(round(denom*b/(a+b)))
                wj=denom-wi
                if min(wi,wj)<=0:
                    continue
                gain=(wi*(best[i]-extra[i])+wj*(best[j]-extra[j]))/denom
                for d in range(nd):
                    weights=np.zeros(4)
                    weights[i]=wi/denom*current_weight[d,i]
                    weights[j]=wj/denom*current_weight[d,j]
                    mass=1.
                    for v in range(4):
                        gain-=weights[v]*miny[d,v]
                        mass-=miny[d,v]
                    for rank in range(4):
                        v=np.argmax(weights)
                        use=max(0.,min(mass,maxy[d,v]-miny[d,v]))
                        gain-=weights[v]*use
                        mass-=use
                        weights[v]=-1e30
                        if mass<=1e-14:
                            break
                if gain>best_bound:
                    best_bound=gain
                    selected=(1<<i)|(1<<j)
                    mix[i]=wi
                    mix[j]=wj
                    if best_bound>threshold:
                        return best_bound,selected
    return best_bound,selected
