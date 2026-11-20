"""Compute recruitment offers from COPT market-participation profits."""
import csv
import hashlib
import itertools
import json
import math
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path

import numpy as np
from market_participation import solve

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'inputs' / 'five.npz'
CACHE = HERE.parent / 'results' / 'physical_coalitions.csv'
TREATMENTS = HERE.parent / 'results' / 'physical_recruitment.json'
LEVELS = (.25, .50, .75, 1.)
BASE = (1, 2, 3, 4, 5)
# Type numbers 1-5 denote the two observed renewable profiles, TG, ESS and DR.
def assets(level, target, mask):
    return tuple(sorted((kind, level if i == 3 else 1.)
                        for i, kind in enumerate((*BASE, target)) if mask & (1 << i)))

def key(group):
    return '|'.join(f'{i}/{power:g}' for i, power in group)

def solve_group(group):
    with np.load(INPUT) as source:
        p = dict(source)
    renew = [kind - 1 for kind, _ in group if kind in (1, 2)]
    ess = [power for kind, power in group if kind == 4]
    tg = sum(kind == 3 for kind, power in group)
    dr = sum(kind == 5 for kind, power in group)
    p['counts'] = (len(renew), tg, len(ess), dr)
    p['renewable'] = p['renewable'][renew]
    for field in ('p_cmax','p_dmax','Esmax','yt_sin','yt_sout','as','socmin','socmax'):
        p[field] = np.repeat(p[field], len(ess))
        if field in ('p_cmax','p_dmax'):
            p[field] *= np.array(ess)
    r = solve(p,tuple(range(1,len(group)+1)),120,1,1e-8)
    if not r['optimal_within_tolerance']:
        raise ValueError((group,r))
    return key(group), (r['lower'],r['upper'])

def bound(coefficients, values):
    score = np.array([values[i] for i in range(64)])
    center = float(coefficients @ score[:,0])
    lo = float(np.where(coefficients>=0,score[:,0],score[:,1]) @ coefficients)
    hi = float(np.where(coefficients>=0,score[:,1],score[:,0]) @ coefficients)
    # Preserve original plotting convention for numerical interval padding.
    pad = 1e-8+1e-5*abs(coefficients).sum()
    return [center, min(center,lo)-pad, max(center,hi)+pad]

def main():
    with (HERE/'inputs'/'five_tight.csv').open(newline='') as stream:
        values={key(tuple((int(i),1.) for i in r['members'].split(','))):
                (float(r['lower']),float(r['upper'])) for r in csv.DictReader(stream)}
    values['']=(0.,0.)
    if CACHE.exists():
        with CACHE.open(newline='') as stream:
            values.update((r['key'],(float(r['lower']),float(r['upper']))) for r in csv.DictReader(stream))
    groups={assets(l,t,m) for l,t,m in itertools.product(LEVELS,(2,4),range(64))}
    missing=sorted((g for g in groups if key(g) not in values),key=len,reverse=True)
    with CACHE.open('w',newline='') as stream, ProcessPoolExecutor(max_workers=5) as pool:
        writer=csv.writer(stream)
        writer.writerow(('key','lower','upper'))
        writer.writerows((k,*v) for k,v in values.items())
        for name, interval in pool.map(solve_group,missing,chunksize=1):
            values[name]=interval
            writer.writerow((name,*interval))
            stream.flush()
    rows={}
    f=np.zeros(64); s=np.zeros(64)
    for m in range(32):
        w=1/(6*math.comb(5,m.bit_count()))
        f[m+32]+=w
        f[m]-=w
    f[32]-=1
    s[[63,31,32]]=[1.,-1.,-1.]
    for target,name in [(2,'PV2'),(4,'ESS')]:
        rows[name]=[]
        for level in LEVELS:
            table={m:values[key(assets(level,target,m))] for m in range(64)}
            rows[name].append([level,bound(f,table),bound(s,table)])
        ref=rows[name][-1][1]
        for level, allocation, contribution in rows[name]:
            a=allocation
            threshold=[1-(.5*ref[0]+.1)/a[0],1-(.5*ref[2]+.1)/a[1],1-(.5*ref[1]+.1)/a[2]]
            next(row for row in rows[name] if row[0]==level).append(threshold)
    record={'treatments':rows,'requested_coalitions':512,'unique_coalitions':len(groups),
            'max_market_profit_absolute_gap':max(b-a for a,b in values.values()),'solver':'COPT',
            'coptpy':version('coptpy'),'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest()}
    TREATMENTS.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    main()

