"""Recompute the analytic two-operator positive control used in the supplement."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Supplement/code"))
from stress_core import two_player_run


if __name__ == "__main__":
    contract = json.loads((HERE / "control_inputs.json").read_bytes())
    result = two_player_run(("control", {}, contract, []))
    print(result["quote"], result["checked_upper"])
    (HERE / "positive_control.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
