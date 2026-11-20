"""Recheck unresolved two-operator rectangles with shared DER allocations."""
import csv
import json
from pathlib import Path

import numpy as np
from numba import njit
from intervals import share, coefficients
from run import inputs_for

HERE = Path(__file__).resolve().parent


@njit
def deviation_lower(coeff, low, high):
    best = -np.inf
    for rule in range(-2, len(coeff)):
        if rule < 0:
            slope, offset = 0., float(rule+2)
        else:
            const, a, b = coeff[rule,:3]
            slope, offset = b/a, (const-1.)/a
        cuts = np.array([low, high, high, high])
        if slope != 0.:
            for k in range(2):
                cuts[k+2] = min(high, max(low, (k-offset)/slope))
            cuts.sort()
        worst = np.inf
        for k in range(3):
            left, right = cuts[k:k+2]
            qleft = min(1., max(0., offset+slope*left))
            qright = min(1., max(0., offset+slope*right))
            # The deviant follows a clipped affine function of the rival quote.
            # Between clipping events all DER shares have extrema at endpoints.
            pay0, pay1 = qleft*coeff[0,5], qright*coeff[0,5]
            for const,a,b,s,fee,_ in coeff:
                y0 = share(const,a,b,qleft,left)
                y1 = share(const,a,b,qright,right)
                w0,w1 = qleft*s-fee,qright*s-fee
                pay0 += fee+min(w0*y0,w0*y1)
                pay1 += fee+min(w1*y0,w1*y1)
            worst = min(worst,pay0,pay1)
        best = max(best,worst)
    return best


@njit
def mixed_bound(game, box, target):
    br0 = deviation_lower(game[0],box[1,0],box[1,1])
    br1 = deviation_lower(game[1],box[0,0],box[0,1])
    n = game.shape[1]
    weights = np.empty(3+n)
    weights[:3] = np.array([0.,1.,0.5])
    ybounds = np.empty((n,2))
    for d in range(n):
        const,a,b = game[0,d,:3]
        ybounds[d,0] = share(const,a,b,box[0,1],box[1,0])
        ybounds[d,1] = share(const,a,b,box[0,0],box[1,1])
        w0 = box[0].mean()*game[0,d,3]-game[0,d,4]
        w1 = box[1].mean()*game[1,d,3]-game[1,d,4]
        weights[3+d] = max(0.,min(1.,w1/(w0+w1))) if abs(w0+w1)>1e-12 else .5
    bound = -np.inf
    for w in weights:
        # Convex combinations of the two deviation gains bound their maximum
        # from below. Couple the payoffs through y_2 = 1 - y_1 for every DER.
        upper = -np.inf
        for x0 in box[0]:
            for x1 in box[1]:
                u = w*x0*game[0,0,5]+(1-w)*x1*game[1,0,5]
                for d in range(n):
                    s0,fee0 = game[0,d,3:5]
                    s1,fee1 = game[1,d,3:5]
                    delta = w*(x0*s0-fee0)-(1.-w)*(x1*s1-fee1)
                    u += w*fee0+(1-w)*x1*s1+max(delta*ybounds[d,0],delta*ybounds[d,1])
                upper = max(upper,u)
        bound = max(bound,w*br0+(1.-w)*br1-upper)
        if bound>target:
            break
    return bound


@njit
def certify(game, limit):
    stack = np.empty((100,2,2))
    stack[0] = np.array([[0.,1.],[0.,1.]])
    pending,n,minimum = 1,0,np.inf
    while pending and n<limit:
        pending -= 1
        box = stack[pending].copy()
        n += 1
        value = mixed_bound(game,box,.001002)
        if value>.001002:
            minimum = min(minimum,value-1e-6)
            continue
        i = int(box[1,1]-box[1,0]>box[0,1]-box[0,0])
        lo,hi = box[i]
        if pending>=99 or hi-lo<1e-12:
            return 0.,n
        mid = .5*(lo+hi)
        stack[pending] = box
        stack[pending,i,1] = mid
        stack[pending+1] = box
        stack[pending+1,i,0] = mid
        pending += 2
    return (0. if pending else minimum),n


def main():
    design = json.loads((HERE/"design.json").read_bytes())
    with (HERE/"outcomes.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    with (HERE/"interval_retries.csv").open("w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["design_row","lower","boxes"])
        for row in rows:
            if row["decision"]!="unresolved":
                continue
            i = int(row["design_row"])
            r = design["configurations"][i]
            game = coefficients(inputs_for(design["markets"][r["market"]],r["key"]))
            lower,n = certify(game,50000)
            writer.writerow([i,lower,n])


if __name__=="__main__":
    main()

