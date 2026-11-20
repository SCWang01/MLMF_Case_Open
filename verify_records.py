"""Check source keys, physical coalition coverage, ancillary results, and plotted data."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

from extract_figure_data import extract

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "small"))
from exact_audit import FrozenGame, full_regrets


def read(name):
    return json.loads((HERE / name).read_bytes())


def verify_ancillary():
    record = read("ancillary/results.json")
    base = read("small/inputs/baseline.json")
    rows = record["results"]
    checked = set()
    for row in rows:
        case, epsilon = row["case"], row["epsilon"]
        mask = row["key"].get("free")
        key = case, epsilon, mask
        if key in checked:
            raise ValueError(f"Duplicate row {key}")
        checked.add(key)
        if case == "control":
            inputs = read("ancillary/control_inputs.json")
        elif case == "five":
            inputs = {**base["inputs"], "epsilon": epsilon}
        elif case == "expiry":
            protected = base["protected_states"][0]["inputs"]
            costs = [list(d) for d in protected["c"]]
            fees = [list(d) for d in protected["compensation"]]
            for d, owner in enumerate(protected["incumbent"]):
                if mask & (1 << d):
                    costs[d] = [0., 0.]
                    fees[owner][d] = 0.
            inputs = {**protected, "epsilon": epsilon, "c": costs, "compensation": fees}
        else:
            raise ValueError(key)
        game = FrozenGame(inputs)
        point = tuple(map(Fraction, row["quote"]))
        if any(not 0 <= q <= 1 for q in point):
            raise ValueError(f"Infeasible quote {key}")
        regret = max(full_regrets(game, point))
        eta = record["settings"][case]["tau"]
        if float(regret) != row["checked_upper"] or row["lower"] > regret:
            raise ArithmeticError(f"Inconsistent bounds {key}")
        if row["decision"] == "numerical_exclusion":
            if row["lower"] <= eta + 1e-6:
                raise ArithmeticError(f"Unverified exclusion {key}")
        elif row["decision"] == "approximate_equilibrium":
            if regret > eta - 1e-6:
                raise ArithmeticError(f"Unverified tolerance {key}")
        else:
            raise ValueError(f"Unresolved decision {key}")
        if row["exact_witness"]:
            witness = tuple(map(Fraction, row["exact_witness"]))
            if any(not 0 <= q <= 1 for q in witness) or max(full_regrets(game, witness)) != 0:
                raise ArithmeticError(f"Invalid exact witness {key}")
    expected = {("control", .3, None)} | {("five", e, None) for e in (.01,.1,1.,10.)}
    expected |= {("expiry", e, mask) for mask in range(32) for e in (.1,10.)}
    if checked != expected:
        raise ValueError("Incomplete ancillary population")


def verify_physical():
    references = read("physical/inputs/reference.json")
    for case, filename, threshold in (("five","physical/five_tight.csv",1.01e-7),
                                      ("twenty","physical_full/coalitions.csv",1.01e-7)):
        with (HERE / filename).open(newline="") as src:
            last = {r["members"]:r for r in csv.DictReader(src)}
        if set(last) != set(references[case]).difference({""}):
            raise ValueError(f"Incomplete physical {case} market")
        for key, r in last.items():
            lower, upper = float(r["lower"]), float(r["upper"])
            if (upper - lower) / max(1., abs(lower)) > threshold or upper < lower - 1e-7:
                raise ArithmeticError(f"Invalid physical bounds {case} {key}")


def main():
    if extract(HERE) != read("figure_data.json"):
        raise ValueError("Figure input or source hash differs")
    verify_physical()
    verify_ancillary()
    all_settings = read("case_setting.json")
    from case_setting import read_contracts
    if all_settings != read_contracts(HERE/"small/inputs/baseline.json", HERE/"large/data/twenty_case.mat"):
        raise ValueError("The case-setting diagram differs from model inputs")
    full = read("large/results.json")
    case_sha = hashlib.sha256((HERE/"large/data/twenty_case.mat").read_bytes()).hexdigest()
    if full["input_sha256"] != case_sha:
        raise ValueError("Unexpected four-operator game")
    for row in full["boxes"]:
        for attempt in range(1, row["attempts"]+1):
            raw = read(f"large/runs/{row['box']}_{attempt}.json")
            if raw["source_sha256"] != case_sha:
                raise ValueError("CCG raw trace input mismatch")
    print("Checked figure inputs, all 69 ancillary COPT records, all 2,074 coalitions and all CCG box attempts.")


if __name__ == "__main__":
    main()

