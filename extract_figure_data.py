"""Collect the numerical figure data from the processed COPT case and sensitivity records."""
from hashlib import sha256
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    sensitivity = Path(os.environ.get("MLMF_SENSITIVITY_OUTPUT", ROOT / "paper_rebuilt/sensitivity")).resolve()
    files = [ROOT / "paper_rebuilt/five/five_data.json", ROOT / "paper_rebuilt/twenty/twenty.json",
             sensitivity / "summary.json"]
    five, twenty, families = [json.loads(p.read_bytes()) for p in files]
    hashes = {(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()):
              sha256(p.read_bytes()).hexdigest() for p in files}
    result = {"solver": "COPT 8.0.6", "case": "Coalition profits obtained from the electricity market participation model.",
              "sources_sha256": hashes, "five": five["figure"], "twenty": twenty,
              "sensitivity": {key: families[key] for key in ("groups", "pv2")}}
    (ROOT / "figure_code/figure_data.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
