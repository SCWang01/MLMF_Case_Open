"""Export COPT market-participation inputs for the standalone supplementary programs."""

import argparse
import csv
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / name).read_bytes())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare(target):
    five = load("physical/five_end_to_end.json")
    twenty = load("alternative/search/search_market.json")["best"]
    profits = {0: 0.0}
    csv_file = ROOT / "physical/five_tight.csv"
    with csv_file.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
        for row in rows:
            mask = sum(1 << (int(member) - 1) for member in row["members"].split(","))
            profits[mask] = max(float(row["lower"]), profits.get(mask, float("-inf")))
    if set(profits) != set(range(32)):
        raise ValueError("The five-DER input requires every coalition")
    target.mkdir(parents=True, exist_ok=True)
    save(target / "five_baseline.json", {"inputs": five["inputs"]})
    save(target / "five_coalitions.json", dict(sorted(profits.items())))
    save(target / "five_twenty_quotes.json", {
        "five": five["best_quote"], "five_regrets": five["exact_audited_regrets"],
        "exclusion_round": five["first_exclusion_round"],
        "twenty": twenty["quote"], "twenty_regrets": twenty["gains"],
    })
    shutil.copyfile(ROOT / "physical/twenty_end_to_end.mat", target / "twenty_case.mat")
    with (target / "five_tight.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["members", "lower", "upper"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    prepare(parser.parse_args().input_dir)
