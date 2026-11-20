"""Independent exact evaluation of the published and recomputed four-VPP candidate quotes."""
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
import json

from exact_four import HERE, K, read_game


def player_optimum(game,quote,v):
    f,c,s,fee,base,eps=game
    # Coefficients are integer-scaled in exact_four. Compute rational affine
    # shares directly from the competitor offers and the simplex envelope.
    cuts={F(0),F(1)}
    envelopes=[]
    for d in range(len(f)):
        rivals=sorted(((1-quote[j])*f[d][j]-c[d][j] for j in range(4) if j!=v),reverse=True)
        lines=[(F(1),F(0))]
        cumulative=F(0)
        for rank,rival in enumerate(rivals,1):
            cumulative+=rival
            den=(rank+1)*eps
            lines.append(((eps+rank*(f[d][v]-c[d][v])-cumulative)/den, F(-rank*f[d][v],den)))
        envelopes.append(lines)
        for (a,b),(aa,bb) in combinations([*lines,(F(0),F(0))],2):
            if b!=bb:
                x=(aa-a)/(b-bb)
                if 0<x<1:
                    cuts.add(x)
    best=None
    def current_payoff(x):
        return (x*base[v]+sum(fee[v][d]+(x*s[v][d]-fee[v][d])*max(F(0),
                   min(a+b*x for a,b in lines)) for d,lines in enumerate(envelopes)))/K
    cut=sorted(cuts)
    for lo,hi in zip(cut,cut[1:]):
        middle=(lo+hi)/2
        a0=F(sum(fee[v]))
        a1=F(base[v])
        a2=F(0)
        for d,lines in enumerate(envelopes):
            a,b=min(lines,key=lambda line:line[0]+middle*line[1])
            if a+b*middle<=0:
                continue
            a0-=fee[v][d]*a
            a1+=s[v][d]*a-fee[v][d]*b
            a2+=s[v][d]*b
        points=[lo,hi]
        if a2<0:
            vertex=-a1/(2*a2)
            if lo<vertex<hi:
                points.append(vertex)
        for x in points:
            profit=(a0+x*(a1+x*a2))/K
            if best is None or best[0]<profit:
                best=(profit,x)
    if best[0]!=current_payoff(best[1]):
        raise ArithmeticError("Upper-envelope optimum failed direct payoff evaluation")
    return dict(regret=best[0]-current_payoff(quote[v]),action=best[1],segments=len(cut)-1)


