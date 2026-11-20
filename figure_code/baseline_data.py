"""BASELINE_DATA Evaluate fixed common ratios against the CCG quote for the baseline figure.

Writes figure_code/baseline_data.json for Fig. 5 (comparison with predetermined ratios):

    five.surplus    - combined coalition surplus of the five-DER baseline (percent reference)
    five.rho_star   - minimum maximum regret, i.e. the maximum regret at the stored CCG quote
    five.grid       - [{c, max_regret, percent}] for the common ratios c = 0, 0.05, ..., 1
    five.profiles   - {name: {label, x, U, regrets, max_regret, percent, der_gain}} for full
                      retention, the midpoint, the best common grid ratio and the CCG quote
    twenty.surplus  - the same three grid / reference fields for the twenty-DER market
    twenty.rho_star
    twenty.grid

Invariant: nothing is written unless the recomputed regrets reproduce the archived
Supplement/results/fixed_quotes.json (absolute tolerance 1e-6) for every profile and grid
row the two files share.

Maintenance: edit PROFILES to change the bar groups of Fig. 5(b). The game inputs and the
archive are only read from Supplement/; this script never modifies them.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUPPLEMENT = HERE.parent / "Supplement"
sys.path.insert(0, str(SUPPLEMENT / "code" / "patched"))
from case20_game import MultiGame, load_game                     # noqa: E402

# ---------- Define data -----------------------------------------------------------
INPUTS   = SUPPLEMENT / "code" / "inputs"
ARCHIVE  = SUPPLEMENT / "results" / "fixed_quotes.json"          # reproduction target
OUTPUT   = HERE / "baseline_data.json"
GRID     = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)         # same grid as the archive
TOL      = 1e-6

# Bar groups of Fig. 5(b), in display order. "best_common" is filled from the grid and
# "ccg" from the stored quote; the archive key is used for the reproduction check.
PROFILES = {
    "full":        {"label": "Full retention", "x": [1.0, 1.0], "archive": "(1,1)"},
    "midpoint":    {"label": "Midpoint",       "x": [0.5, 0.5], "archive": "(0.5,0.5)"},
    "best_common": {"label": "Best common",    "x": None,       "archive": None},
    "ccg":         {"label": "CCG quote",      "x": None,       "archive": "ccg_stored"},
}


class BaselineDataError(ValueError):
    """Raised with a 'baseline_data:<Reason>' identifier."""


def require(condition, reason, message):
    if not condition:
        raise BaselineDataError(f"baseline_data:{reason}: {message}")


# ---------- Load games (mirrors Supplement/code/fixed_quotes.py) -------------------
baseline = json.loads((INPUTS / "five_baseline.json").read_text(encoding="utf-8"))["inputs"]
game5 = MultiGame(**{k: np.asarray(baseline[k], dtype=float)
                     for k in ("f", "c", "base", "contribution", "compensation")},
                  epsilon=float(baseline["epsilon"]),
                  incumbent=np.asarray(baseline["incumbent"], dtype=int))
game20, _ = load_game(str(INPUTS / "twenty_case.mat"))
quotes  = json.loads((INPUTS / "five_twenty_quotes.json").read_text(encoding="utf-8"))
archive = json.loads(ARCHIVE.read_text(encoding="utf-8"))


def surplus(game):
    """Combined cooperative surplus of the current coalitions (all members stay)."""
    own = np.arange(game.v)[:, None] == game.incumbent[None, :]
    return float(np.sum(game.base + np.sum(np.where(own, game.contribution, 0.0), axis=1)))


def grid(game, total):
    rows = []
    for c in GRID:
        worst = float(np.max(game.regrets([float(c)] * game.v)))
        rows.append({"c": float(c), "max_regret": worst, "percent": 100.0 * worst / total})
    return rows


def evaluate(game, x, total):
    x = np.asarray(x, dtype=float)
    regrets = game.regrets(x)
    return {"x": x.tolist(), "U": game.payoffs(x).tolist(), "regrets": regrets.tolist(),
            "max_regret": float(regrets.max()), "percent": float(100.0 * regrets.max() / total),
            "der_gain": float(np.sum(game.shares(x) * game.offers(x)))}


# ---------- Validate: reproduce the archive before deriving anything ---------------
total5, total20 = surplus(game5), surplus(game20)
require(abs(total5 - archive["five_surplus_total"]) <= TOL, "Mismatch",
        f"five-DER surplus {total5} != archived {archive['five_surplus_total']}")
require(abs(total20 - archive["twenty_surplus_total"]) <= TOL, "Mismatch",
        f"twenty-DER surplus {total20} != archived {archive['twenty_surplus_total']}")

grid5, grid20 = grid(game5, total5), grid(game20, total20)
for name, rows, stored in (("five", grid5, archive["five_grid"]["rows"]),
                           ("twenty", grid20, archive["twenty_grid"]["rows"])):
    require(len(rows) == len(stored), "Mismatch", f"{name} grid has {len(rows)} rows, "
            f"archive {len(stored)}")
    for row, ref in zip(rows, stored):
        require(abs(row["c"] - ref["c"]) <= TOL and abs(row["max_regret"] - ref["max_regret"]) <= TOL,
                "Mismatch", f"{name} grid c={row['c']}: {row['max_regret']} vs {ref['max_regret']}")

PROFILES["ccg"]["x"] = [float(z) for z in quotes["five"]]
best = min(grid5, key=lambda r: r["max_regret"])
PROFILES["best_common"]["x"] = [best["c"], best["c"]]

profiles = {}
for key, spec in PROFILES.items():
    row = evaluate(game5, spec["x"], total5)
    if spec["archive"] is not None:
        ref = archive["five_table"][spec["archive"]]
        require(np.allclose(row["regrets"], ref["regrets"], atol=TOL, rtol=0)
                and np.allclose(row["U"], ref["U"], atol=TOL, rtol=0)
                and abs(row["der_gain"] - ref["der_quoted_gain"]) <= TOL,
                "Mismatch", f"profile {key} {spec['x']} does not reproduce archive '{spec['archive']}'")
    profiles[key] = {"label": spec["label"], **row}

rho20 = float(np.max(game20.regrets(np.asarray(quotes["twenty"], dtype=float))))
require(abs(rho20 - max(archive["reproduction"]["twenty"])) <= TOL, "Mismatch",
        f"twenty-DER regret at the CCG quote {rho20} != archived "
        f"{max(archive['reproduction']['twenty'])}")

# ---------- Assemble output ---------------------------------------------------------
output = {
    "five": {"surplus": total5, "rho_star": profiles["ccg"]["max_regret"], "grid": grid5,
             "profiles": profiles},
    "twenty": {"surplus": total20, "rho_star": rho20, "grid": grid20},
    "provenance": {"archive": str(ARCHIVE.relative_to(HERE.parent)).replace("\\", "/"),
                   "inputs": "Supplement/code/inputs", "numpy": np.__version__},
}
OUTPUT.write_text(json.dumps(output, indent=1), encoding="utf-8")
print(f"wrote {OUTPUT}")
for key, row in profiles.items():
    print(f"  {key:12s} x={np.round(row['x'], 4).tolist()} U={np.round(row['U'], 2).tolist()} "
          f"DER={row['der_gain']:.2f} max_regret={row['max_regret']:.2f} ({row['percent']:.2f}%)")
print(f"  five rho*={output['five']['rho_star']:.2f}  best common c={best['c']:.2f} "
      f"({best['max_regret']:.2f}, {best['percent']:.2f}%)")
best20 = min(grid20, key=lambda r: r["max_regret"])
print(f"  twenty rho*={rho20:.2f}  best common c={best20['c']:.2f} "
      f"({best20['max_regret']:.2f}, {best20['percent']:.2f}%)")
