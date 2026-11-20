"""MILP lower bound for minimax regret on a quote box using feasible deviation rules.

The simplex projection KKT conditions and each clipped feasible rule are
piecewise linear. Only products of an operator quote and its DER shares
are relaxed, using the four McCormick facets on locally valid bounds.
"""
from dataclasses import dataclass
import math
import time

import coptpy as cp
import numpy as np


@dataclass
class Term:
    value: object
    lo: float
    hi: float
    # Only populated for projection shares. If y>0 its own price is at most
    # active_q_hi; if y<1 it is at least partial_q_lo.
    active: object = None
    partial_q_lo: float = None
    active_q_hi: float = None


def constant(value):
    value = float(value)
    return Term(value, value, value)


def variable(model, lo, hi):
    if lo == hi:
        return constant(lo)
    return Term(model.addVar(lb=float(lo), ub=float(hi)), float(lo), float(hi))


def pad(*numbers):
    return 1e-11 * max(1., *(abs(x) for x in numbers))


def offer(game, d, v, quote):
    f = float(game.f[d, v])
    c = float(game.c[d, v])
    if quote.lo == quote.hi:
        # Keep conservative offer bounds even when the underlying expression is constant.
        value = f*(1.-quote.lo)-c
        margin = pad(f,c,value)
        return Term(value, value-margin, value+margin)
    lo, hi = f*(1.-quote.hi)-c, f*(1.-quote.lo)-c
    margin = pad(f,c,lo,hi)
    return Term(f*(1.-quote.value)-c, lo-margin, hi+margin)


def maximum(model, terms):
    lo = max(t.lo for t in terms)
    hi = max(t.hi for t in terms)
    contenders = [t for t in terms if t.hi >= lo]
    if len(contenders)==1:
        return contenders[0]
    result = variable(model,lo,hi)
    flags = [model.addVar(vtype=cp.COPT.BINARY) for t in contenders]
    model.addConstr(cp.quicksum(flags)==1.)
    for t,b in zip(contenders,flags):
        model.addConstr(result.value >= t.value)
        model.addConstr(result.value <= t.value+(hi-t.lo)*(1-b))
    return result


def positive(model, expr):
    if expr.lo >= 0.:
        return expr
    if expr.hi <= 0.:
        return constant(0.)
    result = variable(model,0.,expr.hi)
    b = model.addVar(vtype=cp.COPT.BINARY)
    model.addConstr(result.value >= expr.value)
    model.addConstr(result.value <= expr.hi*b)
    model.addConstr(result.value <= expr.value-expr.lo*(1-b))
    return result


def clipped(model, raw):
    if raw.lo>=1.:
        return constant(1.)
    p = positive(model,raw)
    if p.hi<=1.:
        return p
    s = positive(model, Term(1.-p.value,1.-p.hi,1.-p.lo))
    return Term(1.-s.value, max(0.,1.-s.hi), min(1.,1.-s.lo))


def action(model, game, xs, v, rule):
    """Return bounded action and any target share fixed by a capture rule."""
    pivot = None
    if isinstance(rule, dict):
        d = rule["der"]
        offset = rule.get("offset",-1)
        if not isinstance(d,(int,np.integer)) or not 0<=d<game.d or offset not in (-1,1):
            raise ValueError("Invalid capture/departure rule")
        if rule.get("kind","capture" if offset==-1 else "departure") != ("capture" if offset==-1 else "departure"):
            raise ValueError("Policy kind and offset disagree")
        rivals = [offer(game,d,i,x) for i,x in enumerate(xs) if i!=v]
        top = maximum(model,rivals)
        f = float(game.f[d,v])
        intercept = f-float(game.c[d,v])+offset*game.epsilon
        margin = pad(intercept,top.lo,top.hi)/f
        raw = Term((intercept-top.value)/f,
                   (intercept-top.hi)/f-margin, (intercept-top.lo)/f+margin)
        if offset==-1 and raw.lo>=0.:
            pivot = (d,1.)
        elif offset==1 and raw.hi<=1.:
            pivot = (d,0.)
    else:
        if np.ndim(rule)==0:
            return constant(np.clip(float(rule),0.,1.)),None
        coeff = np.asarray(rule,dtype=float)
        if coeff.shape!=(game.v+1,) or not np.all(np.isfinite(coeff)) or coeff[v+1]!=0.:
            raise ValueError("Affine rule must depend only on rivals")
        if not np.any(coeff[1:]):
            return constant(np.clip(coeff[0],0.,1.)),None
        lo = hi = float(coeff[0])
        value = float(coeff[0])
        for b,x in zip(coeff[1:],xs):
            b = float(b)
            value += b*x.value
            lo += min(b*x.lo,b*x.hi)
            hi += max(b*x.lo,b*x.hi)
        margin = pad(np.abs(coeff).sum(),lo,hi)
        raw = Term(value,lo-margin,hi+margin)
    return clipped(model, raw),pivot


def water(offers, eps):
    descending = np.sort(offers)[::-1]
    return float(np.max((np.cumsum(descending)-eps)/np.arange(1,len(offers)+1)))


def projected_share(own, rivals, eps):
    cumulative = np.cumsum(np.sort(rivals)[::-1])
    k = np.arange(1,len(rivals)+1)
    return float(np.clip(np.min((eps+k*own-cumulative)/((k+1)*eps)),0.,1.))


def follower_row(model, game, d, quotes):
    """Exact simplex shares, with dominance screening and bounded complementarity."""
    offers = [offer(game,d,i,q) for i,q in enumerate(quotes)]
    lows = np.array([o.lo for o in offers])
    highs = np.array([o.hi for o in offers])
    eps = game.epsilon
    margin = pad(max(abs(lows)),max(abs(highs)),eps)
    low_water = water(lows,eps)-margin
    high_water = water(highs,eps)+margin
    possible = np.flatnonzero(highs>=low_water)
    result = [constant(0.) for _ in offers]
    if len(possible)==1:
        result[possible[0]]=constant(1.)
        return result
    level = variable(model,low_water,high_water)
    share_pad = 2.*margin/eps
    for i in possible:
        share_lo = max(0.,projected_share(lows[i],np.delete(highs,i),eps)-share_pad)
        share_hi = min(1.,projected_share(highs[i],np.delete(lows,i),eps)+share_pad)
        y = variable(model,share_lo,share_hi)
        slack = eps*y.value+level.value-offers[i].value
        if share_lo>0.:
            y.active = 1.
            model.addConstr(slack==0.)
        else:
            active = model.addVar(vtype=cp.COPT.BINARY)
            y.active = active
            model.addConstr(y.value <= share_hi*active)
            model.addConstr(slack >= 0.)
            model.addConstr(slack <= max(0.,high_water-lows[i]+margin)*(1-active))
        f = float(game.f[d,i])
        intercept = f-float(game.c[d,i])
        q = quotes[i]
        rival_ceilings = np.delete(highs,i)
        rival_floors = np.delete(lows,i)
        y.partial_q_lo = float(np.clip((intercept-eps-max(rival_ceilings)-margin)/f,q.lo,q.hi))
        y.active_q_hi = float(np.clip((intercept-water(rival_floors,eps)+margin)/f,q.lo,q.hi))
        result[i]=y
    model.addConstr(cp.quicksum(result[i].value for i in possible)==1.)
    return result


def mccormick(model, x, y, terms, coefficient, influences):
    if x.lo==x.hi:
        return x.lo*y.value
    if y.lo==y.hi:
        return y.lo*x.value
    value = model.addVar(lb=x.lo*y.lo,ub=x.hi*y.hi)
    model.addConstr(value >= x.lo*y.value+y.lo*x.value-x.lo*y.lo)
    model.addConstr(value >= x.hi*y.value+y.hi*x.value-x.hi*y.hi)
    model.addConstr(value <= x.lo*y.value+y.hi*x.value-x.lo*y.hi)
    model.addConstr(value <= x.hi*y.value+y.lo*x.value-x.hi*y.lo)
    # Local price bounds for positive and nonfull affiliations tighten the
    # global facets. At a pure affiliation the product is already exact.
    if y.active_q_hi is not None and y.active_q_hi < x.hi-1e-9:
        cap = y.active_q_hi
        model.addConstr(value <= cap*y.value)
        model.addConstr(value >= x.value-cap*(1-y.value)-(x.hi-cap)*(1-y.active))
    if y.partial_q_lo is not None and y.partial_q_lo > x.lo+1e-9:
        floor = y.partial_q_lo
        model.addConstr(value <= x.value-floor*(1-y.value))
        full = 0.
        if y.hi>=1.:
            full = model.addVar(vtype=cp.COPT.BINARY)
            model.addConstr(full <= y.value)
        model.addConstr(value >= floor*y.value-(floor-x.lo)*full)
    terms.append((value,x.value,y.value,abs(coefficient),influences))
    return value


def payoff(model, game, v, quote, shares, terms, influences):
    value = float(game.base[v])*quote.value
    for d,y in enumerate(shares):
        s = float(game.contribution[v,d])
        if s:
            value += s*mccormick(model,quote,y,terms,s,influences)
        value += float(game.compensation[v,d])*(1.-y.value)
    return value


def solve_box(game, box, rules, *, seconds=5., threads=1, presolve=0,
              guard=1e-6, env=None):
    """Return a restricted-rule lower bound and a full-BR checked feasible candidate.

    A TIMEOUT can give a dual bound without an incumbent. Zero is returned for
    unreliable statuses or conflicting certificates. Costs are in paper units.
    """
    started = time.perf_counter()
    box = np.asarray(box,dtype=float)
    if box.shape!=(game.v,2) or not np.all(np.isfinite(box)) or np.any(box[:,0]>box[:,1]) or np.any(box<0.) or np.any(box>1.):
        raise ValueError("Box must give one closed subinterval of [0,1] per player")
    if len(rules)!=game.v or seconds<=0 or threads<1 or guard<0:
        raise ValueError("Invalid rules, budget, threads or guard")
    if env is None:
        env = cp.Envr()
    model = env.createModel("linear_minimax_regret")
    for name,value in {"Logging":0,"Threads":int(threads),"TimeLimit":float(seconds),
                       "Presolve":int(presolve),"RelGap":0.,"AbsGap":1e-7,
                       "IntTol":1e-9,"FeasTol":1e-9,"DualTol":1e-9}.items():
        model.setParam(name,value)
    xs = [variable(model,*bounds) for bounds in box]
    rho = model.addVar(lb=0.)
    rows = [follower_row(model,game,d,xs) for d in range(game.d)]
    terms = []
    for v in range(game.v):
        current = payoff(model,game,v,xs[v],[row[v] for row in rows],terms,[v])
        for rule in rules[v]:
            q,pivot = action(model,game,xs,v,rule)
            profile = xs.copy()
            profile[v]=q
            shares = [constant(pivot[1]) if pivot and d==pivot[0] else
                      follower_row(model,game,d,profile)[v] for d in range(game.d)]
            if isinstance(rule,dict):
                influence = [i for i in range(game.v) if i!=v]
            elif np.ndim(rule):
                influence = [i for i,b in enumerate(rule[1:]) if b]
            else:
                influence = []
            model.addConstr(rho >= payoff(model,game,v,q,shares,terms,influence)-current)
    model.setObjective(rho,cp.COPT.MINIMIZE)
    built = time.perf_counter()
    model.solve()
    status = model.status
    raw_bound = lower = 0.
    candidate = upper = None
    usable = status in (cp.COPT.OPTIMAL,cp.COPT.TIMEOUT)
    is_mip = model.bins>0
    if usable:
        if (is_mip and model.hasmipsol) or (not is_mip and model.haslpsol):
            candidate = np.clip([q.lo if q.lo==q.hi else q.value.x for q in xs],box[:,0],box[:,1])
            upper = max(0.,float(max(game.regrets(candidate))))
            candidate = candidate.tolist()
        if is_mip:
            bound = model.bestbnd
        elif status==cp.COPT.OPTIMAL:
            bound = model.objval
        else:
            bound = 0.
        if math.isfinite(bound) and abs(bound)<1e25:
            raw_bound = float(bound)
            lower = max(0.,raw_bound-guard)
    scores = [0.]*game.v
    if candidate is not None:
        def get_value(expr):
            if isinstance(expr,(float,np.number)):
                return float(expr)
            return expr.x if isinstance(expr,cp.Var) else expr.getValue()
        for w,x,y,coef,influences in terms:
            residual = coef*abs(w.x-get_value(x)*get_value(y))
            for i in influences:
                scores[i] += residual/len(influences)
    conflict = upper is not None and lower>upper
    return dict(lower=0. if conflict else lower, raw_bound=raw_bound,
                candidate=candidate,checked_upper=upper,conflict=conflict,split_scores=scores,
                status=status,binary_count=model.bins,rows=model.rows,cols=model.cols,
                build_seconds=built-started,seconds=time.perf_counter()-started)
