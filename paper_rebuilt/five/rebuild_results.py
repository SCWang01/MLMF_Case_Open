"""Reproduce the manuscript's five-DER results from the COPT market participation records."""
import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "large" / "core"))
from case20_game import MultiGame


def market_table(inputs, coalition_path):
    with coalition_path.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != 31 or not all(r["optimal_within_tolerance"] == "True" for r in records):
        raise ValueError("The five-DER market needs 31 solved nonempty coalitions.")
    profits = {r["members"]: float(r["lower"]) for r in records}
    standalone = np.asarray([profits[str(i)] for i in range(1, 6)])
    members = np.asarray(inputs["incumbent"])
    surpluses = []
    for index in range(2):
        mask = members == index
        group = ",".join(str(i + 1) for i in np.flatnonzero(mask))
        surplus = profits[group] - standalone[mask].sum()
        affine_surplus = inputs["base"][index] + np.asarray(inputs["contribution"][index])[mask].sum()
        np.testing.assert_allclose(surplus, affine_surplus, rtol=0, atol=1e-8)
        surpluses.append(float(surplus))
    rows = []
    for d, name in enumerate(("PV1", "PV2", "TG", "ESS", "DR")):
        owner = int(members[d])
        rival = 1 - owner
        remaining = inputs["c"][d][rival] * (5, 4)[rival] / (200, 250)[owner]
        if remaining != round(remaining):
            raise ValueError(f"Invalid contract term for {name}")
        rows.append(dict(der=name, vpp=owner + 1, remaining=int(remaining), standalone=float(standalone[d]),
                         allocated=inputs["f"][d], marginal=[v[d] for v in inputs["contribution"]]))
    return rows, surpluses


def comparison(game, quote, surplus):
    grid = []
    for i in range(21):
        c = i / 20
        regret = float(np.max(game.regrets([c, c])))
        grid.append(dict(c=c, max_regret=regret, percent=100 * regret / surplus))
    best = min(grid, key=lambda r: r["max_regret"])["c"]
    profiles = {}
    for key, label, x in (("full", "Full retention", [1, 1]), ("midpoint", "Midpoint", [0.5, 0.5]),
                          ("best_common", "Best common", [best, best]), ("ccg", "CCG quote", quote)):
        regrets = game.regrets(x)
        maximum = float(np.max(regrets))
        profiles[key] = dict(label=label, x=x, U=game.payoffs(x).tolist(), regrets=regrets.tolist(),
                             max_regret=maximum, percent=100 * maximum / surplus,
                             der_gain=float(np.sum(game.shares(x) * game.offers(x))))
    return dict(surplus=surplus, rho_star=profiles["ccg"]["max_regret"], grid=grid, profiles=profiles)


def derive():
    market = ROOT / "physical" / "five_end_to_end.json"
    coalitions = ROOT / "physical" / "five_tight.csv"
    independent = ROOT / "alternative" / "five" / "market.json"
    result = json.loads(market.read_bytes())
    cert = json.loads(independent.read_bytes())
    inputs = result["inputs"]
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if fingerprint != cert["input_sha256"]:
        raise ValueError("The independently verified game uses different inputs.")
    rows, surpluses = market_table(inputs, coalitions)
    game = MultiGame(**{k: np.asarray(v) if k != "epsilon" else v for k, v in inputs.items()})
    baseline = comparison(game, result["best_quote"], sum(surpluses))
    if abs(baseline["rho_star"] - cert["upper"]) > 1e-6:
        raise ValueError("The quote fails the independent full-domain bound.")
    history, _ = game.iterate(initial=[0.5, 0.5], sweeps=1500)
    regret = np.max(history[:, 5:7], axis=1)
    best = int(np.argmin(regret))
    states = result["protected_states"]
    if states[-1]["exact_equilibrium_regrets"] != ["0", "0"]:
        raise ValueError("The all-expired state needs a verified pure equilibrium.")
    return dict(basis="COPT coalition market participation objectives, allocated using Equal DP and Shapley.",
                source_sha256={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (market, coalitions, independent)},
                table=rows, initial_surpluses=surpluses,
                figure=dict(gains=[dict(expired=r["expired"], vppo=r["regrets"]) for r in states],
                            regret_bounds=[{k: r[k] for k in ("round", "lower", "upper")} for r in result["trace"]],
                            first_exclusion_round=result["first_exclusion_round"]),
                gauss_seidel=dict(sweeps=len(history), min_max_regret=float(regret[best]),
                                  sweep=int(history[best, 0]), profile=history[best, 1:3].tolist()),
                all_expired_equilibrium=[float(Fraction(v)) for v in states[-1]["exact_equilibrium"]],
                full_domain_bound=[cert["lower"], cert["upper"]], baseline=baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path,
                        help="Optionally generate the figure using the paper's render_case_figures.py.")
    args = parser.parse_args()
    data = derive()
    (HERE / "five_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.renderer:
        spec = importlib.util.spec_from_file_location("case_renderer", args.renderer)
        renderer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(renderer)
        renderer.require_font()
        with renderer.plt.rc_context(renderer.STYLE):
            fig = renderer.five_der(data["figure"])
            fig.savefig(HERE / "case_five_ccg.pdf",
                        metadata={"Title": "case_five_ccg", "CreationDate": None, "ModDate": None})
            renderer.plt.close(fig)


if __name__ == "__main__":
    main()

