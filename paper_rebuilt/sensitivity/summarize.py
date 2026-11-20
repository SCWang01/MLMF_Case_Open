"""Join all COPT sensitivity decisions to their independent certificates and figure counts."""
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from sweep import HERE, canonical, games


def records(path, optional=False):
    if optional and not path.is_file():
        return {}
    rows=list(map(json.loads,path.read_text().splitlines()))
    result={canonical(row["key"]):row for row in rows}
    if len(result)!=len(rows):
        raise ValueError(f"Duplicate configuration in {path}")
    return result


def summarize():
    totals=Counter(); ages=Counter(); max_near=0.; n_repaired=0
    for sample in ["base",*map(str,range(10))]:
        name=f"{sample}.jsonl"
        outcomes=records(HERE/"records"/name)
        repairs=records(HERE/"repairs"/name,optional=True)
        intervals=records(HERE/"interval_audits"/name)
        exact=records(HERE/"exact_audits"/name)
        if not set(repairs)<=outcomes.keys():
            raise ValueError(f"Unexpected repair in {sample}")
        outcomes.update(repairs)
        n_repaired+=len(repairs)
        if set(outcomes)!={canonical(k) for k,_ in games(sample)}:
            raise ValueError(f"Incomplete decisions in {sample}")
        within={key for key,r in outcomes.items() if r["decision"]=="within"}
        if set(exact)!=within or set(intervals)-within!=set(outcomes)-within:
            raise ValueError(f"Missing or surplus audits in {sample}")
        for key,row in outcomes.items():
            family=row["key"]["family"]; decision=row["decision"]
            is_exact=False
            if decision=="within":
                if row["upper"]>.000999:
                    raise ValueError(f"Invalid within-tolerance quote: {key}")
                witness=exact[key].get("witness")
                if witness:
                    if len(witness)!=2 or any(not 0<=Fraction(r)<=1 for r in witness):
                        raise ValueError(f"Invalid rational quote: {key}")
                    is_exact=True
                elif (exact[key].get("exact_excluded") and exact[key]["independent_verification"]["verified"]
                      and (HERE/exact[key]["certificate"]).is_file()):
                    max_near=max(max_near,row["upper"])
                else:
                    raise ValueError(f"No exact classification: {key}")
            elif decision!="excluded" or min(row["lower"],intervals[key]["lower"])<=.001001:
                raise ValueError(f"Unverified exclusion: {key}")
            totals[family,decision]+=1
            totals[family,"exact"]+=int(is_exact)
            if family=="ages":
                period=row["key"]["ages"][1]
                ages[period,"total"]+=1; ages[period,"exact"]+=int(is_exact)
    groups=[dict(family=f,total=totals[f,"within"]+totals[f,"excluded"],within=totals[f,"within"],
                 excluded=totals[f,"excluded"],exact=totals[f,"exact"],unresolved=0) for f in ("expiry","ages","resim")]
    pv2=[dict(period=t,total=ages[t,"total"],exact=ages[t,"exact"],unresolved=0) for t in range(4)]
    return dict(groups=groups,pv2=pv2,within=sum(g["within"] for g in groups),excluded=sum(g["excluded"] for g in groups),
                exact=sum(g["exact"] for g in groups),maximum_near_zero_quote_regret=max_near,repaired=n_repaired)


if __name__=="__main__":
    result=summarize()
    (HERE/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(result)
