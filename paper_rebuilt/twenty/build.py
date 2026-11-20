"""Build the twenty-DER manuscript data using the COPT market-participation game."""

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "large/core"))
sys.path.insert(0, str(ROOT / "verification"))
from case20_game import load_game
from check_four_quote import player_optimum
from exact_four import read_game
from global_convergence import global_trace


def load(relative):
    return json.loads((ROOT / relative).read_bytes())


def output(path, records):
    path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build():
    source = ROOT / "physical/twenty_end_to_end.mat"
    game, stored = load_game(source)
    exact_game = read_game(source)
    traces = load("physical/large/results.json")
    refinement = load("alternative/tightened/market.json")
    proof = load("alternative/search/bound_market_97.24961.json")
    search = load("alternative/search/search_market.json")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if {digest, traces["input_sha256"], refinement["sha256"], proof["input_sha256"],
            search["source_sha256"]} != {digest}:
        raise ValueError("Market, CCG traces, and certificates must have the same payoff inputs")
    box_ids = {row["box"] for row in traces["boxes"]}
    if len(box_ids) != len(traces["boxes"]) or box_ids != set(refinement["regions"]) or box_ids != set(proof["box_lowers"]):
        raise ValueError("The local CCG runs and both certificates must cover the same subdomains")

    threshold = traces["tau"] + traces["decision_margin"]
    first_exclusions = []
    for box in traces["boxes"]:
        first = next((row["round"] for row in box["trace"] if row["lower"] > threshold), None)
        if (first is not None) != (box["decision"] == "numerical_exclusion"):
            raise ValueError("A local CCG decision differs from its trace")
        if first is not None:
            first_exclusions.append(first)

    best = search["best"]
    quote = tuple(map(Fraction, best["quote"]))
    gains = [player_optimum(exact_game, quote, v)["regret"] for v in range(game.v)]
    upper = max(gains)
    exact_lower = Fraction(proof["exact_lower"])
    refined_lower = min(row["lower"] for row in refinement["regions"].values())
    if (gains != list(map(Fraction, best["exact_gains"])) or upper != Fraction(best["exact_upper"])
            or refined_lower != refinement["lower"] or float(exact_lower) != proof["global_lower"]
            or not threshold < min(refined_lower, float(exact_lower))
            or max(refined_lower, float(exact_lower)) > float(upper)
            or round(float(exact_lower), 2) != round(float(upper), 2)):
        raise ValueError("Global bounds failed to reproduce the certified minimum")
    full_domain = []
    incumbent = float("inf")
    for row in traces["full_trace"]:
        incumbent = min(incumbent, row["upper"])
        full_domain.append({"round": row["round"], "lower": row["lower"], "upper": incumbent})
    figure = global_trace(ROOT)
    total = float(np.asarray(stored["Parameter"]["IV"]).sum())
    grid = []
    for i in range(21):
        c = Fraction(i, 20)
        profile = (c,) * game.v
        exact_regrets = [player_optimum(exact_game, profile, v)["regret"] for v in range(game.v)]
        np.testing.assert_allclose(game.regrets(np.full(game.v, float(c))),
                                   list(map(float, exact_regrets)), rtol=0, atol=1e-5)
        regret = float(max(exact_regrets))
        grid.append({"c": float(c), "max_regret": regret, "percent": 100 * regret / total})
    baseline = {"surplus": total, "rho_star": float(upper), "grid": grid}
    common = min(grid, key=lambda row: row["max_regret"])
    shares = game.shares(np.array(list(map(float, quote))))
    destinations = shares.argmax(axis=1)
    summary = {
        "source_sha256": digest, "exclusions": len(first_exclusions), "subdomains": len(box_ids),
        "full_domain_lower_bounds": [row["lower"] for row in traces["full_trace"]],
        "full_domain_upper_bounds": [row["upper"] for row in full_domain],
        "minimum_lower": float(exact_lower), "minimum_upper": float(upper),
        "copt_refinement_lower": refined_lower, "quote": list(map(float, quote)),
        "regrets": list(map(float, gains)), "surplus": total,
        "rho_share_percent": 100 * float(upper) / total,
        "best_common": common, "common_multiple": common["max_regret"] / float(upper),
        "initial_contracts": [
            {"members": row["members"].tolist(), "periods": row["member_remain"].tolist(),
             "length": int(row["contract_length"]), "factor": float(row["switching_factor"]),
             "allocation_method": int(row["allocation_method"])} for row in stored["cfg"]["vpp"]],
        "destinations": (destinations + 1).tolist(),
        "minimum_destination_probability": float(shares.max(axis=1).min()),
        "switching_der_ids": (np.flatnonzero(destinations != game.incumbent) + 1).tolist(),
    }
    return figure, baseline, summary


def render(figure_data, renderer, output_dir):
    spec = importlib.util.spec_from_file_location("case_figure_renderer", renderer)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.require_font()
    module.plt.rcParams.update(module.STYLE)
    fig = module.twenty_der(figure_data)
    fig.savefig(output_dir / "case_twenty_ccg.pdf",
                metadata={"Title": "case_twenty_ccg", "CreationDate": None, "ModDate": None})
    module.plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output-dir", type=Path, default=HERE)
    cli.add_argument("--renderer", type=Path, help="Path to the manuscript render_case_figures.py")
    args = cli.parse_args()
    figure, baseline, summary = build()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output(args.output_dir / "twenty.json", figure)
    output(args.output_dir / "baseline_twenty.json", baseline)
    output(args.output_dir / "summary.json", summary)
    if args.renderer is not None:
        render(figure, args.renderer, args.output_dir)
