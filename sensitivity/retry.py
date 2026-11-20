"""Independently retry disputed first-pass COPT bounds with presolve disabled."""
import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from run_copt import copt_check, inputs_for

HERE = Path(__file__).resolve().parent


def first_pass_issues(row, ref, independent):
    label = row["copt"]
    problems = []
    if label == "unresolved":
        problems.append("undecided")
    if float(row["lower"]) > ref["checked_upper"] + 1e-9:
        problems.append("lower_above_published_verified_upper")
    if label == "numerical_exclusion" and independent == "within":
        problems.append("independent_feasible_point")
    if label == "approximate_equilibrium" and independent == "excluded":
        problems.append("interval_countercheck")
    return problems


def main():
    design=json.loads((HERE/"design.json").read_bytes())
    refs=json.loads((HERE/"reference.json").read_bytes())
    with (HERE/"copt_full.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    with (HERE/"outcomes.csv").open(newline="") as f:
        audit={int(r["design_row"]):r["decision"] for r in csv.DictReader(f)}
    if len(rows)!=len(audit) or len(rows)!=len(refs):
        raise ValueError("Incomplete first pass or independent audit")
    tasks=[]
    records={}
    for r in rows:
        i=int(r["design_row"])
        issues=first_pass_issues(r,refs[i],audit[i])
        if issues:
            row=design["configurations"][i]
            tasks.append((i,inputs_for(design["markets"][row["market"]],row["key"]),r["published"],dict(presolve=0)))
            records[i]={"first_pass":r, "reasons":issues}
    with ProcessPoolExecutor(max_workers=4) as executor:
        for i,ref,decision,lo,hi,rounds,sec in executor.map(copt_check,tasks):
            repeat=dict(design_row=i,published=ref,copt=decision,lower=lo,checked_upper=hi,rounds=rounds,seconds=sec)
            records[i]["retry"]=repeat
            problems=first_pass_issues(repeat,refs[i],audit[i])
            if problems:
                repeat["copt"]="unresolved"
                repeat["lower"]=0.0
                records[i]["retry_issues"]=problems
    (HERE/"copt_retries.json").write_text(json.dumps(list(records.values()),indent=2),encoding="utf-8")
    with (HERE/"copt_audited.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=[*rows[0],"presolve"])
        writer.writeheader()
        for r in rows:
            i=int(r["design_row"])
            updated=records[i]["retry"] if i in records else r
            writer.writerow(dict(updated,presolve="off" if i in records else "auto"))
    print(len(records),"COPT retries,",sum("retry_issues" in r for r in records.values()),"still disputed")


if __name__=="__main__":
    main()
