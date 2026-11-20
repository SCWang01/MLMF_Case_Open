"""Check exclusions on the full price square, independently of the COPT master bounds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

from sweep import HERE, ROOT, canonical, games, inputs_for
sys.path.insert(0, str(ROOT / "sensitivity"))
from intervals import coefficients, exclude
from mixed_bounds import certify


def check_batch(tasks):
    output=[]
    for key, inputs in tasks:
        game=coefficients(inputs)
        lower,boxes=exclude(game,1500)
        method="interval"
        if lower <= .001001:
            lower,boxes=certify(game,50000)
            method="weighted_interval"
        output.append(dict(key=key,lower=lower,boxes=boxes,method=method))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=int,default=5)
    p.add_argument("--samples",nargs="+",default=["base",*map(str,range(10))])
    args=p.parse_args()
    dest=HERE/"interval_audits"
    dest.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for sample in args.samples:
            source=HERE/"records"/f"{sample}.jsonl"
            if not source.exists():
                continue
            data={}
            for line in source.read_text().splitlines():
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue
                data[canonical(row["key"])]=row["decision"]
            out=dest/f"{sample}.jsonl"
            done={canonical(json.loads(s)["key"]) for s in out.read_text().splitlines()} if out.exists() else set()
            tasks=[(key,inputs_for(key,market)) for key,market in games(sample)
                   if data.get(canonical(key)) in ("excluded","unresolved") and canonical(key) not in done]
            jobs=[pool.submit(check_batch,tasks[i:i+16]) for i in range(0,len(tasks),16)]
            with out.open("a") as stream:
                for job in as_completed(jobs):
                    for row in job.result():
                        stream.write(json.dumps(row,separators=(",",":"))+"\n")
                    stream.flush()
            print(sample,len(tasks),flush=True)


if __name__=="__main__":
    main()
