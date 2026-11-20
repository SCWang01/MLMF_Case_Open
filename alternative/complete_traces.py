"""Rerun the four roots which have only aggregate rows in the original MILP recording."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from refine import CASES, HERE, ROOT, check_cover, read, run_region, save

NAMES = ("b0000LRLL", "b0000RL", "b0009", "b0013")


def main():
    source = read(ROOT / "large/data/refinement_rules.json")
    candidate = read(ROOT / "verification/candidate_frozen.json")["quote"]
    jobs = [("frozen", name, source["boxes"][name], candidate, 98.4051, 81, 3.0, 120.0, -1, None)
            for name in NAMES]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for row in pool.map(run_region, jobs):
            check_cover(row)
            save(HERE / "replayed_roots" / (row["name"] + ".json"), row)
            print(row["name"], row["lower"], flush=True)

if __name__ == "__main__":
    main()
