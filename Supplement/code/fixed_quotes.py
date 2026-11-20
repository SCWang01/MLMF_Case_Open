"""Evaluate the full-precision COPT game quotes, common-ratio grids and sequential responses."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import scipy

# ---------- Define data ----------------------------------------------------------
# Edit FIVE_QUOTES to change the rows of the main-text table.
FIVE_QUOTES   = {"(0,0)": [0.0, 0.0], "(0.5,0.5)": [0.5, 0.5], "(1,1)": [1.0, 1.0]}
GRID          = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
ROUND_DIGITS  = 4                     # decimal places used in the archived rounding experiment
ESS           = 3                     # DER order: PV1, PV2, TG, ESS, DR
GS_SWEEPS     = 1500                  # Gauss-Seidel sweeps from (0.5, 0.5), as in the archived run
TOLERANCE     = {"eta": 1e-3, "delta": 1e-6, "gap": 1e-3}   # Algorithm 1 settings of the paper
CHECK_TOL     = 5e-3                  # "reproduced to two decimals"
SHARE_TOL     = 1e-12                 # a probability counts as 0 or 1 within this distance
HERE = Path(__file__).resolve().parent
OUTPUT = Path(sys.argv[1]).resolve() if len(sys.argv)==2 else HERE.parent/'results'/'fixed_quotes.json'
FILES = {key: HERE/'inputs'/name for key,name in
         [('baseline','five_baseline.json'),('twenty_mat','twenty_case.mat'),('quotes','five_twenty_quotes.json')]}
sys.path.insert(0,str(HERE/'patched'))
from case20_game import MultiGame, load_game

baseline = json.loads(FILES["baseline"].read_bytes())
inputs   = baseline["inputs"]
game5    = MultiGame(**{k: np.asarray(inputs[k], dtype=float)
                        for k in ("f", "c", "base", "contribution", "compensation")},
                     epsilon=float(inputs["epsilon"]),
                     incumbent=np.asarray(inputs["incumbent"], dtype=int))
saved_quotes = json.loads(FILES["quotes"].read_bytes())
ccg5 = saved_quotes["five"]
ccg20 = saved_quotes["twenty"]
game20, _ = load_game(str(FILES["twenty_mat"]))

def surplus(game):
    """Cooperative surplus I_S of each current coalition (all members stay)."""
    own = np.arange(game.v)[:, None] == game.incumbent[None, :]
    return game.base + np.sum(np.where(own, game.contribution, 0.0), axis=1)


def evaluate(game, x, total):
    """Payoffs, regrets and DER quoted gain at the profile x."""
    x      = np.asarray(x, dtype=float)
    regret = game.regrets(x)
    gain   = float(np.sum(game.shares(x) * game.offers(x)))
    return {"x": x.tolist(), "U": game.payoffs(x).tolist(), "regrets": regret.tolist(),
            "max_regret": float(regret.max()), "percent": float(100 * regret.max() / total),
            "der_quoted_gain": gain}


def grid(game):
    """Max regret of every common ratio on GRID, plus the best grid point."""
    rows = [{"c": float(c), "max_regret": float(np.max(game.regrets([c] * game.v)))} for c in GRID]
    best = min(rows, key=lambda r: r["max_regret"])
    return {"rows": rows, "best": best,
            "at_half": next(r for r in rows if abs(r["c"] - 0.5) < 1e-9)}


total5  = float(surplus(game5).sum())
total20 = float(surplus(game20).sum())
table   = {name: evaluate(game5, x, total5) for name, x in FIVE_QUOTES.items()}
table["ccg_stored"] = evaluate(game5, ccg5, total5)

# ---------- Validate: reproduce archived values before anything is written -------
checks = {"(0.5,0.5)": table["(0.5,0.5)"]["max_regret"], "(1,1)": table["(1,1)"]["max_regret"],
          "ccg": table["ccg_stored"]["max_regret"]}
regrets20 = game20.regrets(np.asarray(ccg20))
if not (np.allclose(regrets20,saved_quotes["twenty_regrets"],atol=1e-6,rtol=0) and
        np.allclose(table["ccg_stored"]["regrets"],saved_quotes["five_regrets"],atol=1e-6,rtol=0)):
    raise ValueError("Reported profiles fail independent regret checks")

# ---------- Derive: rounding effect, switching structure, grids ------------------
rounded  = [round(z, ROUND_DIGITS) for z in ccg5]
rounding = {"stored": {"x": ccg5, "ess_split": game5.shares(np.asarray(ccg5))[ESS].tolist(),
                       "max_regret": table["ccg_stored"]["max_regret"]},
            "rounded": {"x": rounded, "ess_split": game5.shares(np.asarray(rounded))[ESS].tolist(),
                        "max_regret": float(game5.regrets(np.asarray(rounded)).max())}}
y20      = game20.shares(np.asarray(ccg20))
switch   = []
for v in range(game20.v):
    members = [int(d) for d in np.where(game20.incumbent == v)[0]]
    leaving = [d + 1 for d in members if y20[d, v] < 1 - SHARE_TOL]
    entering = [int(d) + 1 for d in np.where(game20.incumbent != v)[0] if y20[d, v] > SHARE_TOL]
    switch.append({"vpp": v + 1, "ratio": ccg20[v], "members": [d + 1 for d in members],
                   "leaving": leaving, "entering": entering})
movers   = sorted({d for s in switch for d in s["leaving"]})
fraction = [{"der": int(d) + 1, "shares": y20[d].tolist()}
            for d in range(game20.d) if np.max(y20[d]) < 1 - SHARE_TOL]

# Sequential best responses: the smallest max regret any sweep reaches.
history, _ = game5.iterate(initial=[0.5, 0.5], sweeps=GS_SWEEPS)
gs_regret  = np.max(history[:, 5:7], axis=1)
gs_best    = int(np.argmin(gs_regret))
gauss      = {"sweeps": int(len(history)), "min_max_regret": float(gs_regret[gs_best]),
              "at_sweep": int(history[gs_best, 0]), "profile": history[gs_best, 1:3].tolist(),
              "final_max_regret": float(gs_regret[-1])}

# Proposition 5 constants (max norm): the projection is 1-Lipschitz, so
# |dy_dv| <= sqrt(V) * max_j F_dj * |dx|_inf / epsilon.
V5, eps5  = game5.v, game5.epsilon
row_max   = game5.f.max(axis=1)
level     = np.abs(game5.base) + np.abs(game5.contribution).sum(axis=1)
spread    = (np.abs(game5.contribution) + game5.compensation) @ row_max
L_U       = float(np.max(level + np.sqrt(V5) * spread / eps5))
positive  = game5.f[game5.f > 0]
L_phi     = float(row_max.max() / positive.min())
Lam       = (2 + L_phi) * L_U
margin    = table["ccg_stored"]["max_regret"] - TOLERANCE["eta"] - 2 * TOLERANCE["delta"] - TOLERANCE["gap"]
radius    = margin / (2 * Lam)
termination = {"L_U": L_U, "L_phi": L_phi, "Lambda": Lam, "margin": margin, "r": radius,
               "round_bound": float((1 + np.ceil(1 / radius)) ** V5 + 1),
               "observed_rounds_to_exclusion": saved_quotes["exclusion_round"]}

# ---------- Assemble output ------------------------------------------------------
result = {
    "reproduction": {"five": checks, "twenty": regrets20.tolist()},
    "five_table": table, "five_surplus_total": total5,
    "five_rounding": rounding, "five_grid": grid(game5),
    "twenty_switch": {"by_vpp": switch, "movers": movers, "n_movers": len(movers),
                      "fractional": fraction},
    "twenty_grid": grid(game20), "twenty_surplus_total": total20,
    "gauss_seidel": gauss, "termination": termination,
    "provenance": {"inputs": {k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in FILES.items()},
                   "numpy": np.__version__, "scipy": scipy.__version__,
                   "python": sys.version.split()[0]},
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
print(json.dumps({"five_table": {k: [round(v["U"][0], 2), round(v["U"][1], 2), round(v["max_regret"], 2),
                                     round(v["percent"], 2), round(v["der_quoted_gain"], 2)]
                                 for k, v in table.items()},
                  "rounding": {k: [np.round(v["ess_split"], 4).tolist(), round(v["max_regret"], 2)]
                               for k, v in rounding.items()},
                  "five_grid_best": result["five_grid"]["best"], "twenty_grid_best": result["twenty_grid"]["best"],
                  "twenty_at_half": result["twenty_grid"]["at_half"], "n_movers": len(movers),
                  "gauss_seidel": gauss, "termination": termination,
                  "fractional": fraction, "totals": [total5, total20]}, indent=1))
