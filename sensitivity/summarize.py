"""Summarize the complete COPT run; verify every row matches the input and publication."""
import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LABEL = {"approximate_equilibrium":"within", "numerical_exclusion":"excluded", "unresolved":"unresolved"}


def summarize():
    design = json.loads((HERE / "design.json").read_bytes())
    configs = design["configurations"]
    refs = json.loads((HERE / "reference.json").read_bytes())
    counts = Counter()
    differences = []
    by_family = Counter()
    seen = set()
    bounds = {}
    with (HERE/"copt_audited.csv").open(newline="", encoding="utf-8") as src:
        for row in csv.DictReader(src):
            index = int(row["design_row"])
            if index in seen or not 0 <= index < len(configs):
                raise ValueError(f"Invalid or duplicate design index {index}")
            seen.add(index)
            family = configs[index]["key"]["family"]
            reference = refs[index]
            if reference["key"] != configs[index]["key"] or reference["decision"] != row["published"]:
                raise ValueError(f"Misaligned publication data on row {index}")
            decision = LABEL[row["copt"]]
            lower = float(row["lower"])
            upper = float(row["checked_upper"]) if row["checked_upper"] else None
            if decision == "within" and (upper is None or upper > 0.000999):
                raise ValueError(f"Missing upper-bound witness for row {index}")
            if decision == "excluded" and (lower <= .001001 or (upper is not None and lower>upper)):
                raise ValueError(f"Invalid numerical exclusion on row {index}")
            counts[family,decision] += 1
            by_family[family] += 1
            bounds[index] = decision
            if row["copt"] != row["published"]:
                differences.append(index)
    if len(seen) != len(configs):
        raise ValueError(f"Incomplete COPT run: {len(seen)} of {len(configs)} inputs")
    groups = [dict(family=group, total=by_family[group],
                   **{label:counts[group,label] for label in LABEL.values()})
              for group in ("expiry","ages","resim")]
    with (HERE/"interval_retries.csv").open(newline="",encoding="utf-8") as file:
        extra = {int(r["design_row"]):float(r["lower"]) for r in csv.DictReader(file)}
    evidence = {}
    with (HERE/"outcomes.csv").open(newline="",encoding="utf-8") as file:
        audit = list(csv.DictReader(file))
        if len(audit) != len(seen) or {int(r["design_row"]) for r in audit} != seen:
            raise ValueError("Incomplete independent audit")
        unresolved = {int(r["design_row"]) for r in audit if r["decision"]=="unresolved"}
        if set(extra) != unresolved:
            raise ValueError("Incomplete second interval pass")
        evidence = Counter()
        for r in audit:
            index = int(r["design_row"])
            decision = "excluded" if extra.get(index,0.)>.001001 else r["decision"]
            evidence[decision,bounds[index]] += 1
    retries = json.loads((HERE/"copt_retries.json").read_bytes())
    first_pass_conflicts = sum("lower_above_published_verified_upper" in r["reasons"] for r in retries)
    first_pass_undecided = sum("undecided" in r["reasons"] for r in retries)
    result = dict(first_pass_conflicting_lower=first_pass_conflicts, first_pass_undecided=first_pass_undecided,
                  presolve_off_retries=len(retries), independent_second_pass=len(extra), groups=groups, discrepancy_rows=differences,
                  method="COPT rule-enhanced CCG plus complete rational best-response checks at unrounded candidate ratios.",
                  independent_audit=[dict(audit=a,copt=b,count=n) for (a,b),n in sorted(evidence.items())])
    (HERE/"copt_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__ == "__main__":
    print(summarize())

