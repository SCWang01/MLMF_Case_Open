"""RENDER_CASE_FIGURES Draw the numerical case figures (Figs. 3-6) in IEEE journal style.

Reads figure_data.json and baseline_data.json and writes four PDFs at the IEEE
single-column width (3.5 in = \\linewidth, so they print at 1:1 and the font sizes
below are final):

    case_five_ccg.pdf    - Fig. 3: (a) five-DER gains at full retention by expired
                           contracts, (b) CCG lower and upper regret bounds by round
    case_twenty_ccg.pdf  - Fig. 4: (a) full-domain upper bound by round (log),
                           (b) global lower bounds against the exclusion threshold
    case_baseline.pdf    - Fig. 5: (a) maximum regret of common posted ratios vs the CCG
                           minimum in both markets, (b) five-DER profits by posting
    case_sensitivity.pdf - Fig. 6: (a) outcome shares by configuration family,
                           (b) states with a verified pure NSE by PV2's remaining term

The manuscript uses Times New Roman. Set --font to select another installed family.
All plotted numbers come from the two data files without filtering or rescaling.

Maintenance: change the look only in "Define style"; change what a figure shows
only in its draw function. Do not edit figure_data.json by hand -- the upstream
extract_figure_data.py identifies its source records by SHA-256; regenerate
baseline_data.json with baseline_data.py.
"""

import argparse
import json
import os
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib import font_manager                              # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402
from matplotlib.ticker import NullLocator                        # noqa: E402


HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
CURRENCY_SYMBOL = json.loads((PAPER / "currency.json").read_text(encoding="utf-8"))["symbol"]

# ---------- Define style ----------------------------------------------------------
# One place for the whole look. Okabe-Ito blue/vermillion were chosen by the
# authors; their gray levels are close, so the second series always also differs
# by hatch, marker or line style (keeps the figures readable in grayscale print).
FONT          = "Times New Roman"
BLUE          = "#0072B2"
VERMILLION    = "#D55E00"
DER_GRAY      = "#9A9A9A"    # third series (DERs) in Fig. 5(b)
GRID_GRAY     = "#C8C8C8"
INK           = "#000000"

FIG_WIDTH_IN  = 3.5          # IEEE column width; keep equal to \linewidth
SIZE_LABEL    = 9.0          # axis labels and panel labels "(a)"
SIZE_TICK     = 8.5
SIZE_NOTE     = 8.0          # legends, value labels, in-plot notes

LW_AXES       = 1.0
LW_DATA       = 2.0
LW_EDGE       = 0.8          # bar outlines, legend frames, hatches
LW_GUIDE      = 1.0          # threshold / event reference lines
MARKER_SIZE   = 6.0
MARKER_EDGE   = 1.5
HATCH_SECOND  = "//"         # texture of every second series
HATCH_THIRD   = ".."         # texture of a third series
BUNDLE_DPI    = 600          # TIFF copies for submission systems (--bundle only)

# Output file and height of each figure; the data key is the figure_data.json key.
FIGURE_SPECS = {
    "five":        {"file": "case_five_ccg.pdf",    "height_in": 2.90},
    "twenty":      {"file": "case_twenty_ccg.pdf",  "height_in": 2.85},
    "baseline":    {"file": "case_baseline.pdf",    "height_in": 3.30},
    "sensitivity": {"file": "case_sensitivity.pdf", "height_in": 3.00},
}

# Display names; edit together with the families written by the sensitivity run.
EXPIRY_LABELS = {"None": "None", "ESS": "ESS only", "All": "All five"}
FAMILY_LABELS = {"expiry": "Coalitions/rules", "ages": "Contract terms",
                 "resim": "Market inputs"}
# Bar groups of Fig. 5(b), in display order; keys are baseline_data.json profile keys.
BASELINE_GROUPS = ("full", "midpoint", "best_common", "ccg")

STYLE = {
    "font.family": FONT, "font.size": SIZE_TICK,
    "mathtext.fontset": "custom", "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK,
    "axes.linewidth": LW_AXES, "axes.labelsize": SIZE_LABEL,
    "axes.labelpad": 2.5, "axes.axisbelow": True,
    "axes.spines.top": True, "axes.spines.right": True,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": SIZE_TICK, "ytick.labelsize": SIZE_TICK,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": LW_AXES, "ytick.major.width": LW_AXES,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.major.pad": 3.0, "ytick.major.pad": 3.0,
    "lines.linewidth": LW_DATA, "lines.markersize": MARKER_SIZE,
    "lines.markeredgewidth": MARKER_EDGE,
    "patch.linewidth": LW_EDGE, "hatch.linewidth": LW_EDGE, "hatch.color": INK,
    "grid.color": GRID_GRAY, "grid.linewidth": 0.5, "grid.linestyle": (0, (3, 2)),
    "legend.fontsize": SIZE_NOTE, "legend.frameon": True, "legend.fancybox": False,
    "legend.edgecolor": INK, "legend.framealpha": 1.0, "legend.borderpad": 0.35,
    "legend.handlelength": 1.6, "legend.handletextpad": 0.45,
    "legend.columnspacing": 1.1, "legend.labelspacing": 0.3,
}


# ---------- Validate environment and data ---------------------------------------
class FigureDataError(ValueError):
    """Raised with a '<Func>:<Reason>' identifier when an input cannot be drawn."""


def require(condition, error_id, message):
    if not condition:
        raise FigureDataError(f"{error_id}: {message}")


def require_font(family):
    try:
        font_manager.findfont(font_manager.FontProperties(family=family),
                              fallback_to_default=False)
    except ValueError as exc:
        raise FigureDataError(f"Matplotlib could not find {family!r}. Install it or use "
                              f"--font \"DejaVu Serif\".") from exc


def require_rounds(rows, first, error_id):
    rounds = [row["round"] for row in rows]
    require(rounds == list(range(first, first + len(rows))), error_id,
            f"rounds must be consecutive from {first}, got {rounds}")
    return rounds


def validate_five(data):
    groups = data["gains"]
    require([g["expired"] for g in groups] == list(EXPIRY_LABELS), "five_der:BadGroups",
            f"expected expiry groups {list(EXPIRY_LABELS)}, got {[g['expired'] for g in groups]}")
    for g in groups:
        require(len(g["vppo"]) == 2 and min(g["vppo"]) >= 0, "five_der:BadGain",
                f"group '{g['expired']}' needs two nonnegative gains, got {g['vppo']}")
    rounds = require_rounds(data["regret_bounds"], 0, "five_der:BadRounds")
    for row in data["regret_bounds"]:
        require(0 <= row["lower"] <= row["upper"] + 1e-6, "five_der:BadBounds",
                f"round {row['round']} has lower {row['lower']} above upper {row['upper']}")
    require(data["first_exclusion_round"] in rounds, "five_der:BadExclusionRound",
            f"first_exclusion_round {data['first_exclusion_round']} is not in {rounds}")


def validate_twenty(data):
    require_rounds(data["full_domain"], 1, "twenty_der:BadFullRounds")
    for row in data["full_domain"]:
        require(row["upper"] > 0, "twenty_der:NonPositiveUpper",
                f"round {row['round']} upper bound {row['upper']} cannot go on a log axis")
    require_rounds(data["cover"], 1, "twenty_der:BadLocalRounds")
    total = data["total_boxes"]
    for row in data["cover"]:
        require(0 <= row["excluded"] <= total, "twenty_der:BadExcluded",
                f"round {row['round']} excludes {row['excluded']} of {total} subdomains")
    require(total - data["cover"][-1]["excluded"] == data["unresolved_boxes"],
            "twenty_der:UnresolvedMismatch",
            f"{total} - {data['cover'][-1]['excluded']} != unresolved {data['unresolved_boxes']}")
    bounds = data["global_bounds"]
    threshold = bounds["decision_threshold"]
    for name in ("copt_refinement", "integer_certificate"):
        require(float(Fraction(str(bounds[name]))) > threshold > 0, "twenty_der:BoundBelowThreshold",
                f"{name} = {bounds[name]} must exceed the positive threshold {threshold}")


def validate_sensitivity(data):
    for row in data["groups"]:
        require(row["family"] in FAMILY_LABELS, "sensitivity:UnknownFamily",
                f"family '{row['family']}' has no display label")
        require(row["unresolved"] == 0 and row["within"] + row["excluded"] == row["total"] > 0,
                "sensitivity:UndecidedConfigurations",
                f"family '{row['family']}' must split {row['total']} cases into within + excluded")
    for row in data["pv2"]:
        require(row["unresolved"] == 0 and 0 <= row["exact"] <= row["total"] and row["total"] > 0,
                "sensitivity:BadPV2State",
                f"PV2 period {row['period']} has {row['exact']} of {row['total']} with "
                f"{row['unresolved']} unresolved")


def validate_baseline(data):
    for market in ("five", "twenty"):
        block = data[market]
        cs = [row["c"] for row in block["grid"]]
        require(cs == sorted(cs) and cs[0] == 0.0 and cs[-1] == 1.0, "baseline:BadGrid",
                f"{market} grid must run from 0 to 1 in increasing c, got {cs[0]}..{cs[-1]}")
        require(min(row["percent"] for row in block["grid"]) > 0 and block["rho_star"] > 0,
                "baseline:NonPositiveRegret", f"{market} regrets must be positive on a log axis")
        require(block["rho_star"] <= min(row["max_regret"] for row in block["grid"]) + 1e-9,
                "baseline:QuoteWorseThanGrid",
                f"{market} CCG minimum {block['rho_star']} exceeds a common-ratio regret")
    for key in BASELINE_GROUPS:
        row = data["five"]["profiles"].get(key)
        require(row is not None and len(row["U"]) == 2 and row["der_gain"] >= 0,
                "baseline:BadProfile", f"profile '{key}' needs two VPPO payoffs and a DER gain")


# ---------- Derive plotting helpers -----------------------------------------------
def new_figure(key):
    """Blank figure of the registered size with a tight constrained layout."""
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIGURE_SPECS[key]["height_in"]),
                     layout="constrained", facecolor="white")
    fig.get_layout_engine().set(w_pad=1.5 / 72, h_pad=2.0 / 72, hspace=0.02, wspace=0.0)
    return fig


def finish_axes(ax, xlabel, panel, ygrid=True):
    """Panel label sits on its own line under the x-label (IEEE convention)."""
    ax.set_xlabel(f"{xlabel}\n({panel})", linespacing=1.55)
    if ygrid:
        ax.grid(axis="y")
    for spine in ax.spines.values():
        spine.set_linewidth(LW_AXES)


def frame_legend(legend):
    legend.get_frame().set_linewidth(LW_EDGE)
    return legend


def note(ax, text, xy, offset, **kwargs):
    """Value label in points-offset from a data point, never clipped by the axes."""
    kwargs.setdefault("fontsize", SIZE_NOTE)
    return ax.annotate(text, xy, xytext=offset, textcoords="offset points",
                       annotation_clip=False, **kwargs)


def boxed():
    """White pad behind text drawn over hatched or colored areas."""
    return {"boxstyle": "square,pad=0.12", "facecolor": "white", "edgecolor": "none"}


# ---------- Assemble figures ------------------------------------------------------
def five_der(data):
    validate_five(data)
    fig = new_figure("five")
    ax_a, ax_b = fig.subplots(2, 1, height_ratios=(1.0, 1.0))

    # (a) Grouped bars: VPPO 1 solid blue, VPPO 2 hatched vermillion.
    series = ({"label": "VPPO 1", "color": BLUE, "hatch": None, "shift": -0.18},
              {"label": "VPPO 2", "color": VERMILLION, "hatch": HATCH_SECOND, "shift": 0.18})
    groups = data["gains"]
    for v, spec in enumerate(series):
        xs = [i + spec["shift"] for i in range(len(groups))]
        ys = [g["vppo"][v] for g in groups]
        ax_a.bar(xs, ys, width=0.34, color=spec["color"], hatch=spec["hatch"],
                 edgecolor=INK, linewidth=LW_EDGE, label=spec["label"], zorder=2)
        for x, y in zip(xs, ys):
            if y > 0:
                note(ax_a, f"{y:.0f}", (x, y), (0, 2), ha="center", va="bottom")
    for i, g in enumerate(groups):
        if max(g["vppo"]) == 0:       # zero-height bars: one label for the pair
            note(ax_a, "0", (i, 0), (0, 2), ha="center", va="bottom")
    ax_a.set(xticks=range(len(groups)), xticklabels=[EXPIRY_LABELS[g["expired"]] for g in groups],
             xlim=(-0.55, len(groups) - 0.45), ylim=(0, 3600), yticks=(0, 1000, 2000, 3000),
             ylabel=f"Gain ({CURRENCY_SYMBOL}/period)")
    ax_a.tick_params(axis="x", length=0)
    frame_legend(ax_a.legend(loc="upper left", ncol=1))
    finish_axes(ax_a, "Expired contracts", "a")

    # (b) Bounds by round; direct labels replace a legend that would cover data.
    rows = data["regret_bounds"]
    rounds = [r["round"] for r in rows]
    lower = [r["lower"] for r in rows]
    upper = [r["upper"] for r in rows]
    ax_b.plot(rounds, upper, color=VERMILLION, ls="-", marker="s", markerfacecolor="white",
              zorder=3)
    ax_b.plot(rounds, lower, color=BLUE, ls=(0, (4, 2)), marker="o", zorder=3)
    note(ax_b, "Upper bound", (0.5, upper[0]), (0, 7), ha="center", va="bottom", color=VERMILLION)
    note(ax_b, "Lower bound", (1.0, lower[0]), (0, 5), ha="center", va="bottom", color=BLUE)
    first = data["first_exclusion_round"]
    ax_b.axvline(first, color=INK, lw=LW_GUIDE, ls=(0, (1, 1.6)), zorder=1)
    note(ax_b, "Pure NSE\nexcluded", (first, 520), (4, 0), ha="left", va="center",
         linespacing=1.1)
    note(ax_b, f"{upper[-1]:.2f}", (rounds[-1], upper[-1]), (6, 0), ha="left", va="center")
    ax_b.set(xticks=rounds, xlim=(rounds[0] - 0.3, rounds[-1] + 0.85), ylim=(-25, 850),
             yticks=(0, 200, 400, 600), ylabel=f"Regret ({CURRENCY_SYMBOL}/period)")
    finish_axes(ax_b, "CCG round", "b")
    fig.align_ylabels((ax_a, ax_b))
    return fig


def twenty_der(data):
    validate_twenty(data)
    fig = new_figure("twenty")
    ax_a, ax_b = fig.subplots(2, 1, height_ratios=(1.0, 0.8))

    # (a) Full-domain upper bound on a log axis.
    rows = data["full_domain"]
    rounds = [r["round"] for r in rows]
    upper = [r["upper"] for r in rows]
    ax_a.plot(rounds, upper, color=BLUE, marker="o", zorder=3)
    note(ax_a, f"{upper[0]:,.2f}", (rounds[0], upper[0]), (-2, 5), ha="left", va="bottom")
    note(ax_a, f"{upper[-1]:.2f}", (rounds[-1], upper[-1]), (-2, 6), ha="right", va="bottom")
    ax_a.set(yscale="log", ylim=(50, 1.2e5), yticks=(1e2, 1e3, 1e4), xticks=rounds,
             xlim=(rounds[0] - 0.4, rounds[-1] + 0.4), ylabel=f"Upper bound\n({CURRENCY_SYMBOL}/period)")
    ax_a.yaxis.set_minor_locator(NullLocator())
    finish_axes(ax_a, "Full-domain CCG round", "a")

    # (b) Dumbbells from the threshold to each global lower bound (log axis). The local
    # subdomain counts (10 of 23 excluded) are stated in the text instead of a panel.
    bounds = data["global_bounds"]
    threshold = bounds["decision_threshold"]
    rows_c = ({"name": "COPT refinement", "value": bounds["copt_refinement"],
               "color": VERMILLION, "marker": "s", "y": 1.0},
              {"name": "Integer certificate", "value": float(Fraction(bounds["integer_certificate"])),
               "color": BLUE, "marker": "D", "y": 0.0})
    ax_b.axvline(threshold, color=INK, lw=LW_GUIDE, ls=(0, (3, 2)), zorder=1)
    for row in rows_c:
        ax_b.hlines(row["y"], threshold, row["value"], colors=row["color"], lw=LW_DATA, zorder=2)
        ax_b.plot(row["value"], row["y"], marker=row["marker"], color=row["color"], ls="",
                  markersize=MARKER_SIZE + 1, zorder=3)
        note(ax_b, row["name"], (threshold * 1.35, row["y"]), (0, 3.5), ha="left", va="bottom")
        note(ax_b, f"{row['value']:.2f}", (row["value"], row["y"]), (6, 0), ha="left",
             va="center", color=row["color"])
    note(ax_b, f"Threshold {threshold:g}", (threshold * 1.35, -0.42), (0, 0), ha="left",
         va="center")
    ax_b.set(xscale="log", xlim=(4e-4, 1.2e3), xticks=(1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2),
             ylim=(-0.75, 1.7), yticks=())
    ax_b.xaxis.set_minor_locator(NullLocator())
    finish_axes(ax_b, f"Global lower bound ({CURRENCY_SYMBOL}/period)", "b", ygrid=False)
    return fig


def baseline(data):
    validate_baseline(data)
    fig = new_figure("baseline")
    ax_a, ax_b = fig.subplots(2, 1, height_ratios=(1.0, 1.0))

    # (a) Max regret of a common posted ratio, relative to each market's surplus.
    markets = ({"key": "five", "label": "Five DERs", "color": BLUE, "ls": "-", "marker": "o"},
               {"key": "twenty", "label": "Twenty DERs", "color": VERMILLION, "ls": (0, (4, 2)),
                "marker": "s"})
    for m in markets:
        block = data[m["key"]]
        cs = [row["c"] for row in block["grid"]]
        pct = [row["percent"] for row in block["grid"]]
        best = min(block["grid"], key=lambda row: row["max_regret"])
        # The minimum goes in the legend: beside the marker it would sit on a CCG line.
        ax_a.plot(cs, pct, color=m["color"], ls=m["ls"], zorder=3,
                  label=f"{m['label']} (min {best['percent']:.2f}%)")
        ax_a.plot(best["c"], best["percent"], marker=m["marker"], color=m["color"], ls="",
                  markerfacecolor="white", zorder=4)
        star = 100.0 * block["rho_star"] / block["surplus"]
        ax_a.axhline(star, color=m["color"], lw=LW_GUIDE, ls=(0, (1, 1.6)), zorder=2)
        note(ax_a, f"CCG {star:.2f}%", (1.0, star), (-2, 2.5), ha="right", va="bottom",
             color=m["color"])
    ax_a.set(yscale="log", ylim=(0.12, 160), yticks=(0.1, 1, 10, 100),
             yticklabels=("0.1", "1", "10", "100"), xlim=(0, 1),
             xticks=(0, 0.2, 0.4, 0.6, 0.8, 1.0), ylabel="Max. regret\n(% of surplus)")
    ax_a.yaxis.set_minor_locator(NullLocator())
    # Outside the axes: inside, the legend would cover the rising twenty-DER curve.
    frame_legend(ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=2,
                             borderaxespad=0.0))
    finish_axes(ax_a, "Common posted ratio $c$", "a")

    # (b) Estimated profit per period of each party under four five-DER postings.
    profiles = data["five"]["profiles"]
    parties = ({"label": "VPPO 1", "color": BLUE, "hatch": None, "value": lambda p: p["U"][0]},
               {"label": "VPPO 2", "color": VERMILLION, "hatch": HATCH_SECOND,
                "value": lambda p: p["U"][1]},
               {"label": "DERs", "color": DER_GRAY, "hatch": HATCH_THIRD,
                "value": lambda p: p["der_gain"]})
    width = 0.26
    for k, party in enumerate(parties):
        xs = [i + (k - 1) * width for i in range(len(BASELINE_GROUPS))]
        ys = [party["value"](profiles[key]) for key in BASELINE_GROUPS]
        ax_b.bar(xs, ys, width=width * 0.92, color=party["color"], hatch=party["hatch"],
                 edgecolor=INK, linewidth=LW_EDGE, label=party["label"], zorder=2)
        for x, y in zip(xs, ys):
            if y == 0:                # a zero-height bar would read as missing
                note(ax_b, "0", (x, 0), (0, 2), ha="center", va="bottom")
    ticklabels = [f"{profiles[key]['label']}\n$\\rho$ = {profiles[key]['max_regret']:.0f}"
                  for key in BASELINE_GROUPS]
    ax_b.set(xticks=range(len(BASELINE_GROUPS)), xticklabels=ticklabels,
             xlim=(-0.55, len(BASELINE_GROUPS) - 0.45), ylim=(0, 4600),
             yticks=(0, 1000, 2000, 3000, 4000), ylabel=f"Profit ({CURRENCY_SYMBOL}/period)")
    ax_b.tick_params(axis="x", length=0)
    frame_legend(ax_b.legend(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=3,
                             borderaxespad=0.0))
    finish_axes(ax_b, "Posted ratios (five DERs)", "b")
    fig.align_ylabels((ax_a, ax_b))
    return fig


def sensitivity(data):
    validate_sensitivity(data)
    fig = new_figure("sensitivity")
    # Subfigures keep panel (a)'s long row labels from widening panel (b)'s margin.
    sub_a, sub_b = fig.subfigures(2, 1, height_ratios=(1.08, 1.0))
    ax_a = sub_a.subplots()
    ax_b = sub_b.subplots()

    # (a) Share of configurations within tolerance vs numerically excluded.
    parts = ({"key": "within", "label": "Within tolerance", "color": BLUE, "hatch": None},
             {"key": "excluded", "label": "Numerical exclusion", "color": VERMILLION,
              "hatch": HATCH_SECOND})
    groups = data["groups"]
    for i, row in enumerate(groups):
        left = 0.0
        for part in parts:
            width = 100.0 * row[part["key"]] / row["total"]
            ax_a.barh(i, width, left=left, height=0.62, color=part["color"], hatch=part["hatch"],
                      edgecolor=INK, linewidth=LW_EDGE, label=part["label"] if i == 0 else None,
                      zorder=2)
            note(ax_a, f"{width:.1f}", (left + width / 2, i), (0, 0), ha="center", va="center",
                 bbox=boxed())
            left += width
    ax_a.set(yticks=range(len(groups)), ylim=(len(groups) - 0.45, -0.55), xlim=(0, 100),
             xticks=(0, 25, 50, 75, 100),
             yticklabels=[f"{FAMILY_LABELS[r['family']]}\n(n = {r['total']:,})" for r in groups])
    ax_a.tick_params(axis="y", length=0)
    frame_legend(ax_a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=2,
                             borderaxespad=0.0))
    finish_axes(ax_a, "Share of configurations (%)", "a", ygrid=False)

    # (b) Verified pure NSE among the 500 contract states of each PV2 term.
    pv2 = data["pv2"]
    periods = [r["period"] for r in pv2]
    for row in pv2:
        share = 100.0 * row["exact"] / row["total"]
        ax_b.bar(row["period"], share, width=0.55, color=BLUE, edgecolor=INK,
                 linewidth=LW_EDGE, zorder=2)
        note(ax_b, f"{row['exact']}/{row['total']}", (row["period"], share), (0, 2),
             ha="center", va="bottom")
    ax_b.set(xticks=periods, xticklabels=["0 (expired)" if p == 0 else str(p) for p in periods],
             xlim=(periods[0] - 0.5, periods[-1] + 0.5), ylim=(0, 122), yticks=(0, 50, 100),
             ylabel="States with\npure NSE (%)")
    finish_axes(ax_b, "Remaining contract periods of PV2", "b")
    return fig


# ---------- Command line ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=HERE / "figure_data.json")
    parser.add_argument("--baseline-data", type=Path, default=HERE / "baseline_data.json",
                        help="output of baseline_data.py (Fig. 5)")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "Figs")
    parser.add_argument("--bundle", type=Path, default=None,
                        help="also write SVG and TIFF copies here (the manuscript uses the PDFs)")
    parser.add_argument("--font", default=os.environ.get("MLMF_FIGURE_FONT", FONT),
                        help="font family for labels and equations (default: manuscript font)")
    args = parser.parse_args()

    require_font(args.font)
    plt.rcParams.update(STYLE)
    plt.rcParams.update({"font.family": args.font, "mathtext.rm": args.font,
                         "mathtext.it": f"{args.font}:italic", "mathtext.bf": f"{args.font}:bold"})
    data = json.loads(args.data.read_text(encoding="utf-8"))
    base = json.loads(args.baseline_data.read_text(encoding="utf-8"))
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.bundle is not None:
        args.bundle.mkdir(exist_ok=True, parents=True)
    drawers = {"five": (five_der, data["five"]), "twenty": (twenty_der, data["twenty"]),
               "baseline": (baseline, base), "sensitivity": (sensitivity, data["sensitivity"])}
    for key, (draw, source) in drawers.items():
        fig = draw(source)
        target = args.output_dir / FIGURE_SPECS[key]["file"]
        # Fixed metadata keeps the PDFs byte-stable across re-runs.
        fig.savefig(target, metadata={"Title": target.stem, "CreationDate": None,
                                      "ModDate": None})
        print(f"wrote {target}")
        if args.bundle is not None:
            stem = args.bundle / target.stem
            fig.savefig(stem.with_suffix(".svg"), metadata={"Date": None})
            fig.savefig(stem.with_suffix(".tiff"), dpi=BUNDLE_DPI,
                        pil_kwargs={"compression": "tiff_lzw"})
            print(f"wrote {stem}.svg and {stem}.tiff")
        plt.close(fig)


if __name__ == "__main__":
    main()
