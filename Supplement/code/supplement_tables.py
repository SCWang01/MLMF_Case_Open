"""SUPPLEMENT_TABLES Write the supplement's numeric LaTeX tables from the result JSON files.

Run:  python -E -P supplement_tables.py <paper folder>

Writes <paper>/Supplement/tables/:
    dynamics.tex   - per-window expectations for both charge schedules
    grids.tex      - max regret of every common ratio for both markets
    epsilon.tex    - Algorithm 1 decisions at epsilon = 100 and 1000
    constants.tex  - Proposition 4 constants and the Gauss-Seidel record (macros)

Invariant: every number is read from Supplement/results/*.json; none is typed by hand.
Maintenance: change formatting here; rerun after any result file changes.
"""

import json
import sys
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path

# ---------- Validate -------------------------------------------------------------
if len(sys.argv) != 2:
    raise SystemExit("supplement_tables:Usage python supplement_tables.py <paper folder>")
PAPER   = Path(sys.argv[1]).resolve()
RESULTS = PAPER / "Supplement" / "results"
TABLES  = PAPER / "Supplement" / "tables"
FILES   = {name: RESULTS / f"{name}.json" for name in ("dynamics", "fixed_quotes", "epsilon_stress")}
for name, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(f"supplement_tables:MissingInput {name} -> {path}")
data = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in FILES.items()}


# ---------- Derive -----------------------------------------------------------------
def dynamics_table():
    head = ("\\begin{tabular}{@{}rrrrrrrrrr@{}}\n\\toprule\n"
            "Window & $P_\\eta$ & $\\mathbb E\\rho$ & $\\mathbb E x_1$ & $\\mathbb E x_2$ & "
            "Cash$_1$ & Cash$_2$ & DER gain & Switches & $P_{\\text{empty}}$\\\\\n\\midrule\n")
    body = []
    labels = {"time_varying": "Time-variant switching cost $\\gamma_{o(d)}r_d$", "flat": "Flat switching cost 500"}
    for name, label in labels.items():
        body.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{label}}}}}\\\\\n")
        for w in data["dynamics"]["schedules"][name]["windows"]:
            body.append(f"{w['window']} & {w['p_pure']:.2f} & {w['max_regret']:.2f} & {w['x'][0]:.2f} & "
                        f"{w['x'][1]:.2f} & {w['vppo_cash'][0]:.2f} & {w['vppo_cash'][1]:.2f} & "
                        f"{w['der_gain']:.2f} & {w['switches']:.2f} & {w['p_empty_vpp']:.2f}\\\\\n")
    return head + "".join(body) + "\\bottomrule\n\\end{tabular}\n"


def grids_table():
    five, twenty = data["fixed_quotes"]["five_grid"]["rows"], data["fixed_quotes"]["twenty_grid"]["rows"]
    rows = "".join(f"{a['c']:.2f} & {a['max_regret']:.2f} & {b['max_regret']:.2f}\\\\\n"
                   for a, b in zip(five, twenty))
    return ("\\begin{tabular}{@{}rrr@{}}\n\\toprule\nCommon ratio $c$ & Five DERs & Twenty DERs\\\\\n"
            "\\midrule\n" + rows + "\\bottomrule\n\\end{tabular}\n")


def regret_bound(value, rounding):
    """Round a displayed regret bound outwards, retaining tiny nonzero bounds."""
    number = Decimal(str(value))
    if number and abs(number) < Decimal("0.01"):
        quantum = Decimal(1).scaleb(number.adjusted() - 2)
        rounded = number.quantize(quantum, rounding=rounding)
        exponent = rounded.adjusted()
        return rf"{rounded.scaleb(-exponent):.2f}\times10^{{{exponent}}}"
    return f"{number.quantize(Decimal('0.01'), rounding=rounding):.2f}"


def epsilon_table():
    rows = []
    for r in data["epsilon_stress"]["results"]:
        lower = regret_bound(r["lower"], ROUND_FLOOR)
        upper = regret_bound(r["checked_upper"], ROUND_CEILING)
        probs = "; ".join(f"{p[0]:.2f}" for p in r["affiliation_probabilities"])
        rows.append(f"{r['epsilon']:g} & {r['decision'].replace('_', ' ')} & "
                    f"$[{lower},{upper}]$ & "
                    f"$({r['quote'][0]:.2f},{r['quote'][1]:.2f})$ & {probs}\\\\\n")
    return ("\\begin{tabular}{@{}rllll@{}}\n\\toprule\n$\\epsilon$ & Decision & Bracket & Quote & "
            "$y_{d1}$ (PV1; PV2; TG; ESS; DR)\\\\\n\\midrule\n" + "".join(rows)
            + "\\bottomrule\n\\end{tabular}\n")


def sci(value, digits=2):
    """LaTeX scientific notation, e.g. 6.05\\times10^{-8}."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}\\times10^{{{int(exponent)}}}"


def constants():
    t, g = data["fixed_quotes"]["termination"], data["fixed_quotes"]["gauss_seidel"]
    q = data["fixed_quotes"]
    macros = {"LU": sci(t["L_U"]), "Lphi": f"{t['L_phi']:.2f}", "Lam": sci(t["Lambda"]),
              "rad": sci(t["r"]), "bound": sci(t["round_bound"]),
              "gsMin": f"{g['min_max_regret']:.2f}", "gsSweep": str(g["at_sweep"]),
              "gsFinal": f"{g['final_max_regret']:.2f}",
              "gsXa": f"{g['profile'][0]:.2f}", "gsXb": f"{g['profile'][1]:.2f}",
              "nMovers": str(q["twenty_switch"]["n_movers"]),
              "stateTV": str(len(data["dynamics"]["schedules"]["time_varying"]["states"])),
              "stateFlat": str(len(data["dynamics"]["schedules"]["flat"]["states"]))}
    return "".join(f"\\newcommand{{\\S{k}}}{{{v}}}\n" for k, v in macros.items())


# ---------- Assemble output ------------------------------------------------------
TABLES.mkdir(parents=True, exist_ok=True)
outputs = {"dynamics.tex": dynamics_table(), "grids.tex": grids_table(),
           "epsilon.tex": epsilon_table(), "constants.tex": constants()}
for name, text in outputs.items():
    (TABLES / name).write_text(text, encoding="utf-8")
    print("wrote", TABLES / name)
