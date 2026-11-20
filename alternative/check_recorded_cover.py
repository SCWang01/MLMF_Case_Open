"""Verify the original 23-root summary and an independently completed set of node traces."""
import json
from pathlib import Path

from refine import HERE, ROOT, check_cover

def main():
    read = lambda p: json.loads(p.read_bytes())
    summary = read(HERE / "refinement/frozen.json")
    boxes = read(ROOT / "large/data/twenty_boxes.json")
    if set(summary["regions"]) != set(boxes):
        raise ValueError("Missing regional summary")
    trace_minima = {}
    replayed = []
    for name, box in boxes.items():
        path = HERE / "refinement/frozen" / (name + ".json")
        original = path.is_file()
        if not original:
            path = HERE / "replayed_roots" / (name + ".json")
            replayed.append(name)
        row = read(path)
        if row["case"] != "frozen" or row["name"] != name or row["steps"][0]["bounds"] != box:
            raise ValueError(f"Unexpected region {name}")
        check_cover(row)
        if original and row["lower"] != summary["regions"][name]["lower"]:
            raise ValueError(f"Recorded bound differs at {name}")
        trace_minima[name] = row["lower"]
    recorded = min(row["lower"] for row in summary["regions"].values())
    if min(trace_minima.values()) != recorded or recorded != summary["lower"] or recorded <= .001001:
        raise ArithmeticError("Cover does not certify the stated global lower bound")
    print(len(boxes), "complete trees; lower =", recorded, "; roots with fresh traces:", ", ".join(replayed))

if __name__ == "__main__":
    main()
