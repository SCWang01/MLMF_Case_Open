"""Replay exact nonexistence certificates and independently re-evaluate the COPT near-zero quotes."""
import csv
from fractions import Fraction
import json
from pathlib import Path
from verify_cases import FrozenGame, check_candidate, full_regrets, verify


def main():
    folder = Path(__file__).resolve().parent
    read = lambda name: json.loads((folder / name).read_bytes())
    baseline = read("../small/inputs/baseline.json")
    verify(FrozenGame(baseline["inputs"]), read("certificates/baseline.json"))
    cases = read("near_zero_cases.json")
    records = {r["id"]:r for r in read("../sensitivity/near_zero_copt.json")}
    configs = read("../sensitivity/design.json")["configurations"]
    with (folder / "../sensitivity/copt_audited.csv").open(newline="") as fp:
        rows = {int(r["design_row"]):r for r in csv.DictReader(fp)}
    if len(cases) != 15 or {c["id"] for c in cases} != set(records):
        raise ValueError("Incomplete near-zero certificate or COPT quote set")
    legacy_max = sweep_max = replay_max = 0.
    for case in cases:
        game = FrozenGame(case["inputs"])
        verify(game, read("certificates/" + case["id"] + ".json"))
        legacy = check_candidate(game, case["candidate_audit"])
        record = records[case["id"]]
        i = record["design_row"]
        if record["key"] != case["key"] or configs[i]["key"] != case["key"]:
            raise ValueError(f"Unexpected design key for {case['id']}")
        point = tuple(map(Fraction, record["quote"]))
        if any(not 0 <= v <= 1 for v in point):
            raise ArithmeticError("Infeasible COPT quote")
        gains = full_regrets(game,point)
        regret = max(gains)
        if list(map(str,gains)) != record["gains"] or regret != Fraction(record["exact_checked_upper"]):
            raise ValueError("COPT exact quote audit mismatch")
        sweep = float(rows[i]["checked_upper"])
        if record["full_sweep_upper"] != sweep or rows[i]["copt"] != "approximate_equilibrium":
            raise ValueError("COPT full-sweep record mismatch")
        if not 0 < legacy <= Fraction("8.47e-7") or not (0 < sweep <= .000999 and 0 < regret <= Fraction(".000999")):
            raise ArithmeticError(f"Invalid approximate witness for {case['id']}")
        legacy_max=max(legacy_max,float(legacy))
        sweep_max=max(sweep_max,sweep)
        replay_max=max(replay_max,float(regret))
    print(f"Baseline and 15 exact nonexistence certificates verified. Historical witness maximum {legacy_max:.12g}. "
          f"COPT full-sweep maximum {sweep_max:.12g}. Checked COPT quote maximum {replay_max:.12g}.")

if __name__ == "__main__":
    main()

