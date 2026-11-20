"""Four-player search using an exact-in-one-dimension response envelope.

Exploration uses only game coefficients. Previous quotes enter a separate,
labelled local refinement after all independent evolutionary runs complete.
All reported best quotes are audited with the rational full-interval oracle.
"""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit
from scipy.optimize import differential_evolution, minimize
from scipy.stats import qmc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/"large/core"))
sys.path.insert(0,str(ROOT/"verification"))
from case20_game import load_game
from exact_four import read_game
from check_four_quote import player_optimum


@njit
def payoff_piece(mid, intercept, slope, base, contribution, fee):
    # Clipped lower envelope of the three projection pieces and the cap 1.
    payoff = mid * base
    derivative = base
    quadratic = 0.
    for d in range(len(fee)):
        y, dy = 1., 0.
        for k in range(3):
            z = intercept[d,k] - slope[d,k] * mid
            if z < y:
                y, dy = z, -slope[d,k]
        if y <= 0.:
            y, dy = 0., 0.
        weight = mid*contribution[d]-fee[d]
        payoff += fee[d]+weight*y
        derivative += contribution[d]*y+weight*dy
        quadratic += contribution[d]*dy
    return payoff, derivative, quadratic


@njit
def gains(x, f, c, base, contribution, fee, eps):
    nd, nv = f.shape
    out = np.empty(nv)
    for v in range(nv):
        event = np.empty(4*nd+2)
        event[:2] = (0.,1.)
        n = 2
        b = np.empty((nd,3))
        a = np.empty((nd,3))
        for d in range(nd):
            rivals = np.empty(3)
            j = 0
            for w in range(4):
                if w != v:
                    rivals[j] = (1-x[w])*f[d,w]-c[d,w]
                    j += 1
            # Fixed-size sort avoids numerical tolerances on near ties.
            r0,r1,r2 = rivals
            if r0 < r1: r0,r1 = r1,r0
            if r0 < r2: r0,r2 = r2,r0
            if r1 < r2: r1,r2 = r2,r1
            ordered = (r0,r1,r2)
            cumulative = 0.
            active, level_sum = 0, 0.
            for k in range(3):
                quoted = ordered[k]
                # When the follower adds rival k to its support.
                threshold = eps+(k+1)*quoted-cumulative
                if cumulative-k*quoted < eps:
                    active += 1
                    level_sum += quoted
                    z = (f[d,v]-c[d,v]-threshold)/f[d,v]
                    if 0. < z < 1.:
                        event[n] = z
                        n += 1
                cumulative += quoted
                b[d,k] = (eps+(k+1)*(f[d,v]-c[d,v])-cumulative)/((k+2)*eps)
                a[d,k] = (k+1)*f[d,v]/((k+2)*eps)
            z = (f[d,v]-c[d,v]-(level_sum-eps)/active)/f[d,v]
            if 0. < z < 1.:
                event[n] = z
                n += 1
        knots = np.sort(event[:n])
        best = -np.inf
        for j in range(1,n):
            lo, hi = knots[j-1:j+1]
            if lo == hi: continue
            mid = .5*(lo+hi)
            u,du,qq = payoff_piece(mid,b,a,base[v],contribution[v],fee[v])
            h = (hi-lo)/2
            best = max(best,u+h*(abs(du)+qq*h))
            if qq < 0.:
                dx = -du/(2*qq)
                if -h < dx < h:
                    best = max(best,u+dx*(du+qq*dx))
        current = payoff_piece(x[v],b,a,base[v],contribution[v],fee[v])[0]
        out[v] = best-current
    return out


def exact_audit(data,q):
    quote = tuple(map(Fraction,q))
    rows = [player_optimum(data,quote,i) for i in range(4)]
    worst = max(r["regret"] for r in rows)
    return dict(quote=list(map(float,quote)),gains=[float(r["regret"]) for r in rows],
                exact_gains=list(map(lambda r:str(r["regret"]),rows)),
                exact_upper=str(worst),upper=float(worst))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",required=True,choices=("frozen","dispatch"))
    args = parser.parse_args()
    src = ROOT / ("large/data/twenty_case.mat" if args.case=="frozen" else "physical/twenty_end_to_end.mat")
    game,_ = load_game(src)
    params = tuple(np.ascontiguousarray(v) for v in (game.f,game.c,game.base,game.contribution,game.compensation))+(game.epsilon,)
    data = read_game(src)
    # Check the fast oracle at corners and independent space-filling points.
    validate = np.vstack([np.zeros(4),np.ones(4),qmc.Sobol(4,scramble=True,seed=647).random_base2(5)])
    error = max(float(max(abs(gains(x,*params)-game.regrets(x)))) for x in validate)
    if error > 5e-6:
        raise RuntimeError(f"One-dimensional best-response check failed: {error}")
    def objective(q):
        return float(max(gains(q,*params)))
    def polish(x):
        # Estimate a local epigraph on each side of nearby active-set kinks.
        # Use multiple radii and re-anchor at the best feasible quote found.
        winner=x.copy()
        val=objective(winner)
        for r in (1e-3,1e-4,1e-5,1e-6,1e-7):
            for ids in (np.arange(4),np.flatnonzero((winner>1e-8)&(winner<1.-1e-8))):
                if not len(ids): continue
                anchor=winner.copy()
                def lift(z):
                    q=anchor.copy()
                    q[ids]=np.clip(z,0.,1.)
                    return q
                center=winner[ids].copy()
                dx=np.where(center>.5,-r,r)
                simplex=np.vstack([center,center+np.diag(dx)])
                fit=minimize(lambda z:objective(lift(z)),center,method="Nelder-Mead",
                             options=dict(initial_simplex=simplex,maxiter=360,xatol=1e-13,fatol=2e-9))
                if fit.fun<val:
                    winner=lift(fit.x)
                    val=objective(winner)
        return winner
    clock=time.perf_counter()
    rounds=[]
    bounds=[(0.,1.)]*4
    for k in range(5):
        # Alternate the whole cube and each lower face independently of the
        # old quotes. Include many starts near boundaries for the narrow regions.
        mode=["cube","faces","low-biased","cube","faces"][k]
        initial=qmc.Sobol(4,scramble=True,seed=2400+k).random_base2(6)
        if mode=="faces":
            for i in range(len(initial)):
                initial[i,i%4]=0.
        if mode=="low-biased":
            initial=initial**2
        fit=differential_evolution(objective,bounds,seed=9900+k,init=initial,maxiter=155,
                                   tol=1e-10,atol=5e-8,polish=False,recombination=.75)
        q=polish(fit.x)
        rounds.append(dict(seed=9900+k,starts=mode,evaluations=fit.nfev,search_upper=fit.fun,
                           quote=q.tolist(),polished_upper=objective(q)))
    independent=min(rounds,key=lambda r:r["polished_upper"])
    old=json.loads((ROOT/"verification"/("candidate_"+args.case+".json")).read_bytes())["quote"]
    warm=polish(np.array(old))
    checked_independent=exact_audit(data,independent["quote"])
    checked_local=exact_audit(data,warm)
    old_check=exact_audit(data,old)
    best=min([old_check,checked_local,checked_independent],key=lambda row:Fraction(row["exact_upper"]))
    record=dict(source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),case=args.case,
                max_oracle_absolute_error=error,oracle_check_points=len(validate),explorations=rounds,
                independent=checked_independent,previous=old_check,refined_previous=checked_local,
                best=best,elapsed_seconds=time.perf_counter()-clock)
    (HERE/f"search_{args.case}.json").write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(args.case,checked_independent["upper"],old_check["upper"],checked_local["upper"],flush=True)


if __name__=="__main__":
    main()

