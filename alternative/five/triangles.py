"""Global two-operator minimax search with explicit DER clipping and COPT LP bounds."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from fractions import Fraction as F
from functools import lru_cache
import gzip
import hashlib
from heapq import heappop, heappush
import json
from pathlib import Path
import sys
from time import perf_counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "small"))
from exact_audit import FrozenGame, full_regrets

ZERO, ONE = F(0), F(1)
WEIGHT_SCALE = 10**10
SQUARE = [(ZERO,ZERO),(ONE,ZERO),(ONE,ONE),(ZERO,ONE)]


def clamp(x):
    return min(ONE,max(ZERO,x))


def midpoint(a,b):
    return tuple((x+y)/2 for x,y in zip(a,b))


def line_value(line, p):
    return line[0]+line[1]*p[0]+line[2]*p[1]


def polygon_area(poly):
    return sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(poly,poly[1:]+poly[:1]))/2


def clip_polygon(poly, line, sign):
    output=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        x,y=sign*line_value(line,a),sign*line_value(line,b)
        if x>=0:
            output.append(a)
        if x*y<0:
            t=x/(x-y)
            output.append(tuple(u+(v-u)*t for u,v in zip(a,b)))
    return output


def partition(game):
    polygons=[SQUARE]
    for a,b,c in game.difference:
        for boundary in (-game.epsilon,game.epsilon):
            line=(a-boundary,b,c)
            next_polygons=[]
            for poly in polygons:
                signed=[line_value(line,p) for p in poly]
                if min(signed)<0<max(signed):
                    next_polygons.extend(clip_polygon(poly,line,sign) for sign in (-1,1))
                else:
                    next_polygons.append(poly)
            polygons=next_polygons
    if sum(polygon_area(p) for p in polygons)!=1:
        raise ArithmeticError("Response regions do not cover the unit square")
    return polygons


def active_coefficients(game, line, point):
    a,b,c=line
    value=line_value(line,point)
    if value>=game.epsilon:
        return ZERO,ZERO,ZERO
    if value<=-game.epsilon:
        return ONE,ZERO,ZERO
    e=2*game.epsilon
    return (game.epsilon-a)/e,-b/e,-c/e


def actions(game, polygons, player):
    """Endpoints, saturation kinks, and stationary own quotes on every open region."""
    rival=1-player
    pool={(ZERO,ZERO),(ONE,ZERO)}
    for line in game.difference:
        for threshold in (-game.epsilon, game.epsilon):
            pool.add(((threshold-line[0])/line[1+player],-line[1+rival]/line[1+player]))
    for poly in polygons:
        center=tuple(sum(z[i] for z in poly)/len(poly) for i in range(2))
        linear,quadratic,cross=game.base[player],ZERO,ZERO
        for d,line in enumerate(game.difference):
            y=active_coefficients(game,line,center)
            linear+=game.sc[player][d]*y[0]+game.fees[player][d]*y[1+player]
            quadratic+=game.sc[player][d]*y[1+player]
            cross+=game.sc[player][d]*y[1+rival]
        if quadratic<0:
            pool.add((-linear/(2*quadratic),-cross/(2*quadratic)))
    return sorted((a,b) for a,b in pool if (a,b) in ((ZERO,ZERO),(ONE,ZERO)) or
                  (max(a,a+b)>0 and min(a,a+b)<1))


def interior_root(cuts, intercept, slope, target, lo, hi):
    if slope:
        r=(target-intercept)/slope
        if lo<r<hi:
            cuts.add(r)


def quad_value(poly, x):
    return poly[0]+x*(poly[1]+x*poly[2])


class PayoffRule:
    """An exact piecewise quadratic for a feasible deviation against rival ratio r."""
    def __init__(self,game,player,action):
        self.action=action
        a,b=action
        clips={ZERO,ONE}
        for target in (ZERO,ONE):
            interior_root(clips,a,b,target,ZERO,ONE)
        cuts=set(clips)
        for lo,hi in zip(sorted(clips),sorted(clips)[1:]):
            q=a+b*(lo+hi)/2
            k,m=(ZERO,ZERO) if q<=0 else ((ONE,ZERO) if q>=1 else (a,b))
            for line in game.difference:
                intercept=line[0]+line[1+player]*k
                slope=line[1+player]*m+line[2-player]
                for threshold in (-game.epsilon,game.epsilon):
                    interior_root(cuts,intercept,slope,threshold,lo,hi)
        self.knots=sorted(cuts)
        self.polys=[]
        for lo,hi in zip(self.knots,self.knots[1:]):
            r=(lo+hi)/2
            q=a+b*r
            k,m=(ZERO,ZERO) if q<=0 else ((ONE,ZERO) if q>=1 else (a,b))
            own=clamp(q)
            point=(own,r) if player==0 else (r,own)
            base,grow,fee,fee_grow=game.base[player],ZERO,ZERO,ZERO
            for d,line in enumerate(game.difference):
                z=active_coefficients(game,line,point)
                offset=z[0]+z[1+player]*k
                slope=z[2-player]+z[1+player]*m
                base+=game.sc[player][d]*offset
                grow+=game.sc[player][d]*slope
                fee+=game.fees[player][d]*offset
                fee_grow+=game.fees[player][d]*slope
            self.polys.append((k*base+fee,m*base+k*grow+fee_grow,m*grow))

    def index(self,r):
        return min(bisect_right(self.knots,r)-1,len(self.polys)-1)

    def value(self,r):
        return quad_value(self.polys[self.index(r)],r)

    @lru_cache(maxsize=30000)
    def minorant(self,lo,hi):
        # Translate the endpoint chord downward until it lies below every
        # quadratic piece on [lo,hi]. Stationary points are checked exactly.
        if lo==hi:
            return self.value(lo),ZERO
        slope=(self.value(hi)-self.value(lo))/(hi-lo)
        values=[]
        for i in range(self.index(lo),self.index(hi)+1):
            a,b,c=self.polys[i]
            b-=slope
            l=max(lo,self.knots[i])
            h=min(hi,self.knots[i+1])
            values.extend((a+l*(b+l*c),a+h*(b+h*c)))
            if c>0:
                r=-b/(2*c)
                if l<r<h:
                    values.append(a+r*(b+r*c))
        return min(values),slope


def bernstein(values):
    a,b,c,mab,mbc,mca=values
    return (a,b,c,2*mab-(a+b)/2,2*mbc-(b+c)/2,2*mca-(c+a)/2)


def split(tri):
    a,b,c=tri
    sides=[(a,b,c),(b,c,a),(c,a,b)]
    a,b,c=max(sides,key=lambda t:sum((x-y)**2 for x,y in zip(t[0],t[1])))
    m=midpoint(a,b)
    return (a,m,c),(m,b,c)


def read_data(case):
    baseline=json.loads((ROOT/"small/inputs/baseline.json").read_bytes())
    if case=="frozen":
        return baseline["inputs"],baseline["protected_states"]
    rebuilt=json.loads((ROOT/"physical/five_end_to_end.json").read_bytes())["inputs"]
    economic={k:rebuilt[k] for k in ("base","f","contribution")}
    states=[{"expired":s["expired"],"inputs":dict(s["inputs"],**economic)} for s in baseline["protected_states"]]
    return rebuilt,states


class Search:
    def __init__(self,inputs):
        self.game=FrozenGame(inputs)
        self.polygons=partition(self.game)
        self.roots=[]
        for poly in self.polygons:
            for j in range(1,len(poly)-1):
                tri=(poly[0],poly[j],poly[j+1])
                if polygon_area(list(tri))>0:
                    self.roots.append(tri)
        if sum(polygon_area(list(t)) for t in self.roots)!=1:
            raise ArithmeticError("Invalid triangulation")
        self.rules=[[PayoffRule(self.game,i,action) for action in actions(self.game,self.polygons,i)]
                    for i in range(2)]
        self.best=None
        self.quote=None
        self.lp_count=0

    @lru_cache(maxsize=None)
    def response(self,i,r):
        values=[rule.value(r) for rule in self.rules[i]]
        idx=max(range(len(values)),key=values.__getitem__)
        return idx,values[idx]

    @lru_cache(maxsize=None)
    def current(self,p):
        y=self.game.responses(p)
        return tuple(self.game.payoff_from_y(i,p,y) for i in range(2))

    def check_quote(self,p):
        value=max(self.response(i,p[1-i])[1]-self.current(p)[i] for i in range(2))
        if self.best is None or value<self.best:
            self.best,self.quote=value,p
        return value

    def coefficients(self,tri,ids):
        a,b,c=tri
        points=(*tri,midpoint(a,b),midpoint(b,c),midpoint(c,a))
        rows=[]
        for i,j in ids:
            ratios=[p[1-i] for p in tri]
            k,m=self.rules[i][j].minorant(min(ratios),max(ratios))
            rows.append(bernstein([k+m*p[1-i]-self.current(p)[i] for p in points]))
        return rows

    def bound(self,tri,tol,lp_env,cp):
        for p in tri:
            self.check_quote(p)
        ids=[]
        for i in range(2):
            r=[p[1-i] for p in tri]
            r.extend(((min(r)+max(r))/2,sum(r)/3))
            ids.extend((i,j) for j in sorted({self.response(i,z)[0] for z in r}))
        rows=self.coefficients(tri,ids)
        j=max(range(len(rows)),key=lambda k:min(rows[k]))
        baseline=min(rows[j])
        if baseline>=self.best-tol or len(rows)==1:
            return baseline,[[*ids[j],1]]
        # Select simplex weights that maximize the lowest Bernstein coefficient.
        # Their integer-normalized values, not an LP bound, are used as evidence.
        active=[k for k,row in enumerate(rows) if max(row)>=baseline]
        best_id=ids[j]
        ids=[ids[k] for k in active]
        rows=[rows[k] for k in active]
        j=ids.index(best_id)
        scale=max(abs(val-baseline) for row in rows for val in row)
        if not scale or len(rows)==1:
            return baseline,[[*ids[j],1]]
        model=lp_env.createModel()
        model.setParam("Logging",0)
        model.setParam("Threads",1)
        w=[model.addVar(lb=0.,ub=1.) for _ in ids]
        z=model.addVar(lb=-cp.COPT.INFINITY)
        model.addConstr(cp.quicksum(w)==1)
        for col in zip(*rows):
            model.addConstr(z<=cp.quicksum(float((v-baseline)/scale)*weight for v,weight in zip(col,w)))
        model.setObjective(z,cp.COPT.MAXIMIZE)
        model.solve()
        self.lp_count+=1
        if model.status!=cp.COPT.OPTIMAL:
            return baseline,[[*ids[j],1]]
        weights=[max(0,round(WEIGHT_SCALE*var.x)) for var in w]
        total=sum(weights)
        if total<=0:
            return baseline,[[*ids[j],1]]
        certified=min(sum(w*v for w,v in zip(weights,col))/total for col in zip(*rows))
        if certified<=baseline:
            return baseline,[[*ids[j],1]]
        return certified,[[*idx,w] for idx,w in zip(ids,weights) if w]

    def solve(self,tol,max_nodes):
        import coptpy as cp
        env=cp.Envr()
        heap=[]
        trace=[]
        self.check_quote((F(1,2),F(1,2)))
        def push(tri,root,path):
            lower,witness=self.bound(tri,tol,env,cp)
            heappush(heap,(lower,root,path,tri,witness))
        for root,tri in enumerate(self.roots):
            push(tri,root,"")
        def snapshot(iteration):
            trace.append(dict(splits=iteration,lower=float(heap[0][0]),upper=float(self.best),
                              gap=float(self.best-heap[0][0]),lp_solves=self.lp_count,leaves=len(heap)))
        iteration=0
        snapshot(0)
        while self.best-heap[0][0]>tol:
            if iteration>=max_nodes:
                raise RuntimeError(f"Node budget reached: [{float(heap[0][0])}, {float(self.best)}]")
            _,root,path,tri,_=heappop(heap)
            for k,child in enumerate(split(tri)):
                push(child,root,path+str(k))
            iteration+=1
            if iteration & (iteration-1)==0:
                snapshot(iteration)
        if trace[-1]["splits"]!=iteration:
            snapshot(iteration)
        exact=max(full_regrets(self.game,self.quote))
        if exact!=self.best:
            raise ArithmeticError("Piecewise response envelope disagrees with independent full best responses")
        return dict(quote=list(map(str,self.quote)),leaves=[[root,p,w] for _,root,p,_,w in heap]),trace

    def verify(self,cert):
        by_root=[{} for _ in self.roots]
        for k,path,w in cert["leaves"]:
            if path in by_root[k]:
                raise ValueError("Repeated leaf")
            by_root[k][path]=w
        minimum=None
        for root,leaves in zip(self.roots,by_root):
            stack=[("",root)]
            seen=set()
            branches={p[:n] for p in leaves for n in range(len(p))}
            while stack:
                path,tri=stack.pop()
                if path not in leaves:
                    if path not in branches:
                        raise ValueError("Incomplete triangle cover")
                    stack.extend((path+str(k),child) for k,child in enumerate(split(tri)))
                    continue
                witness=leaves[path]
                if any(not isinstance(w,int) or w<=0 or i not in (0,1) or not 0<=j<len(self.rules[i])
                       for i,j,w in witness):
                    raise ValueError("Invalid deviation witness")
                rows=self.coefficients(tri,[(i,j) for i,j,_ in witness])
                total=sum(w for _,_,w in witness)
                bound=min(sum(t[2]*v for t,v in zip(witness,col))/total for col in zip(*rows))
                minimum=bound if minimum is None else min(minimum,bound)
                seen.add(path)
            if len(seen)!=len(leaves):
                raise ValueError("Overlapping triangle leaves")
        point=tuple(map(F,cert["quote"]))
        if not all(0<=x<=1 for x in point):
            raise ValueError("Invalid candidate")
        upper=max(full_regrets(self.game,point))
        if minimum>upper:
            raise ArithmeticError("Inconsistent global bounds")
        return minimum,upper


def controls(states):
    results=[]
    for state in states:
        game=FrozenGame(state["inputs"])
        regions=partition(game)
        point=(ONE,ONE)
        gains=[]
        for i in range(2):
            best=max(PayoffRule(game,i,a).value(ONE) for a in actions(game,regions,i))
            gains.append(best-game.payoff(i,point))
        if tuple(gains)!=full_regrets(game,point):
            raise ArithmeticError("Incorrect control best response")
        results.append(dict(expired=state["expired"],regrets=list(map(float,gains))))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case",choices=["frozen","dispatch","both"],default="both")
    parser.add_argument("--verify",action="store_true",help="replay certificates with rational arithmetic, without COPT")
    parser.add_argument("--tolerance",default="0.000000000001")
    parser.add_argument("--max-splits",type=int,default=20000)
    args=parser.parse_args()
    tolerance=F(args.tolerance)
    if tolerance<=0 or args.max_splits<1:
        raise ValueError("Positive tolerance and split budget required")
    for case in (["frozen","dispatch"] if args.case=="both" else [args.case]):
        start=perf_counter()
        inputs,states=read_data(case)
        digest=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(",",":")).encode()).hexdigest()
        search=Search(inputs)
        path=HERE/(case+"_certificate.json.gz")
        if not args.verify:
            cert,trace=search.solve(tolerance,args.max_splits)
            cert["absolute_tolerance"]=str(tolerance)
            cert["input_sha256"]=digest
            path.write_bytes(gzip.compress(json.dumps(cert,separators=(",",":")).encode(),mtime=0))
        else:
            cert=json.loads(gzip.decompress(path.read_bytes()))
        if cert["input_sha256"]!=digest:
            raise ValueError("Certificate inputs do not match")
        lower,upper=search.verify(cert)
        if upper-lower>F(cert["absolute_tolerance"]):
            raise ArithmeticError("Certified gap exceeds target")
        print(case,float(lower),float(upper),len(cert["leaves"]),flush=True)
        if not args.verify:
            summary=dict(case=case,method="Two-player response-region triangulation and COPT LP mixing",
                         input_sha256=digest,quote=list(map(float,map(F,cert["quote"]))),lower=float(lower),upper=float(upper),
                         exact_lower=str(lower),exact_upper=str(upper),gap=float(upper-lower),
                         polygons=len(search.polygons),initial_triangles=len(search.roots),
                         rules_per_player=list(map(len,search.rules)),trace=trace,controls=controls(states),
                         seconds_including_verification=perf_counter()-start)
            (HERE/(case+".json")).write_text(json.dumps(summary,indent=2),encoding="utf-8")
        # Instance methods are memoized for one market at a time.
        Search.response.cache_clear()
        Search.current.cache_clear()
        PayoffRule.minorant.cache_clear()


if __name__=="__main__":
    main()

