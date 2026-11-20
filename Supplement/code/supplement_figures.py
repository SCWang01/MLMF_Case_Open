"""Draw the contract-expiry, multi-window dynamics, and physical recruitment figures.

Run: python supplement_figures.py [--dynamics-only]

Writes:
    <paper>/Figs/expiry_subsets.pdf        - equilibrium decisions for 32 expiry subsets
    <paper>/Figs/dynamics.pdf              - multi-window dynamics
    <paper>/Figs/physical_recruitment.pdf  - effect of storage power on recruitment

Invariant: plotted numbers come only from COPT result files under
Supplement/results/dynamics.json; nothing is re-estimated here.
Maintenance: styling follows the archive's make_figures.PAPER_STYLE; edit the
"Define data" block to change sizes or colors.
"""
import argparse
import json
import sys
from pathlib import Path

# ---------- Define data ----------------------------------------------------------
TEAL, ORANGE, GRAY, DARK = "#007C83", "#B86623", "#B5BCC4", "#49545D"
COLUMN_MM, FULL_MM       = 88, 180
SCHEDULE_STYLE           = {"time_varying": ("-", "Time-variant switching cost"), "flat": ("--", "Flat switching cost")}

# ---------- Validate -------------------------------------------------------------
PAPER = Path(__file__).resolve().parents[2]
CURRENCY_SYMBOL = json.loads((PAPER / "currency.json").read_text(encoding="utf-8"))["symbol"]
DYNAMICS = PAPER/'Supplement/results/dynamics.json'
import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.lines import Line2D                          # noqa: E402
from matplotlib.patches import Patch, Rectangle              # noqa: E402
import numpy as np                                           # noqa: E402
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix', 'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5, 'xtick.labelsize': 7.7, 'ytick.labelsize': 7.7, 'legend.fontsize': 7.7, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'legend.frameon': False, 'lines.linewidth': 1.1, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})


# ---------- Derive: expiry subsets at column width --------------------------------
def expiry_subsets():
    expiry_decisions = {int(k): v for k, v in json.loads((DYNAMICS.parent / "expiry_subsets.json").read_bytes()).items()}
    fig = plt.figure(figsize=(COLUMN_MM / 25.4, 1.62))
    ax  = fig.add_axes([0.135, 0.255, 0.85, 0.50])
    masks    = sorted(range(32), key=lambda m: (m.bit_count(), m))
    labels   = {"approximate_equilibrium": "Within tolerance",
                "numerical_exclusion": "Numerical exclusions", "unresolved": "Unresolved"}
    styles   = {"Within tolerance": (TEAL, ""), "Numerical exclusions": (GRAY, "///"),
                "Unresolved": ("#E7B35A", "..")}
    outcomes = [labels[expiry_decisions[m]] for m in masks]
    for col, (mask, outcome) in enumerate(zip(masks, outcomes)):
        for bit in range(5):
            ax.add_patch(Rectangle((col - 0.5, bit - 0.5), 1, 1, linewidth=0.4, edgecolor="white",
                                   facecolor=DARK if mask & (1 << bit) else "#F1F3F5"))
        color, hatch = styles[outcome]
        ax.add_patch(Rectangle((col - 0.5, 4.65), 1, 0.85, facecolor=color, hatch=hatch,
                               edgecolor="#64707A", linewidth=0.25))
    ax.set(xlim=(-0.5, 31.5), ylim=(5.5, -0.5), yticks=range(6),
           yticklabels=["PV1", "PV2", "TG", "ESS", "DR", "Outcome"])
    special = [masks.index(m) for m in [0, 8, 31]]
    ax.set_xticks(special, ["None", "ESS only", "All"])
    ax.tick_params(axis="both", length=0, pad=1.5, labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for n in range(6):
        cols   = [i for i, m in enumerate(masks) if m.bit_count() == n]
        lo, hi = cols[0] - 0.5, cols[-1] + 0.5
        ax.plot([lo, hi], [-0.8, -0.8], color="#68727A", lw=0.5, clip_on=False)
        ax.text((lo + hi) / 2, -1.05, str(n), ha="center", va="bottom", fontsize=7)
        if n:
            ax.axvline(lo, color="white", lw=1.2)
    for col in special:
        ax.add_patch(Rectangle((col - 0.5, -0.5), 1, 6, fill=False, edgecolor="#202B33", linewidth=0.6))
    fig.text(0.56, 0.975, "Number of expired contracts (dark cells = expired)", fontsize=7,
             ha="center", va="top")
    fig.legend(handles=[Patch(facecolor=c, hatch=h, edgecolor=DARK, linewidth=0.3,
                              label=f"{outcomes.count(name)} {name.lower()}")
                        for name, (c, h) in styles.items() if name in outcomes],
               loc="lower center", bbox_to_anchor=(0.56, -0.01), ncol=2, fontsize=7,
               handlelength=1.2, handletextpad=0.35, columnspacing=1.0)
    return fig


# ---------- Derive: dynamics figure ------------------------------------------------
def dynamics():
    data = json.loads(DYNAMICS.read_text(encoding="utf-8"))["schedules"]
    fig  = plt.figure(figsize=(COLUMN_MM / 25.4, 1.95))
    ax   = fig.add_axes([0.11, 0.30, 0.355, 0.52])
    bx   = fig.add_axes([0.625, 0.30, 0.355, 0.52])
    fig.text(0.012, 0.94, "a", fontsize=9.5, weight="bold")
    fig.text(0.11, 0.94, "Posted ratios", fontsize=8.5, weight="bold")
    fig.text(0.515, 0.94, "b", fontsize=9.5, weight="bold")
    fig.text(0.625, 0.94, "Per-period outcomes", fontsize=8.5, weight="bold")
    for name, (style, _) in SCHEDULE_STYLE.items():
        windows = data[name]["windows"]
        t   = np.array([w["window"] for w in windows])
        x   = np.array([w["x"] for w in windows])
        der = np.array([w["der_gain"] for w in windows]) / 1000
        vp1 = np.array([w["vppo_cash"][0] for w in windows]) / 1000
        ax.plot(t, x[:, 0], style, color=TEAL, marker="o", ms=2.2)
        ax.plot(t, x[:, 1], style, color=ORANGE, marker="^", ms=2.2)
        bx.plot(t, vp1, style, color=TEAL, marker="o", ms=2.2)
        bx.plot(t, der, style, color=DARK, marker="s", ms=2.2)
    ax.set(xlim=(0.6, 10.4), ylim=(-0.03, 1.03), xticks=[1, 4, 7, 10], yticks=[0, 0.5, 1],
           xlabel="Window", ylabel="Expected ratio")
    bx.set(xlim=(0.6, 10.4), ylim=(-0.15, 4.8), xticks=[1, 4, 7, 10], yticks=[0, 2, 4],
           xlabel="Window", ylabel=f"Amount (thousand {CURRENCY_SYMBOL})")
    for panel in (ax, bx):
        panel.tick_params(pad=2)
    ax.text(3.0, 0.72, "VPPO 1", color=TEAL, ha="center", fontsize=7)
    ax.text(10.3, 0.20, "VPPO 2", color=ORANGE, ha="right", fontsize=7)
    bx.text(3.0, 3.25, "VPPO 1 cash", color=TEAL, ha="center", fontsize=7)
    bx.text(10.3, 0.55, "DER gain", color=DARK, ha="right", fontsize=7)
    fig.legend(handles=[Line2D([], [], color=DARK, ls=s, lw=1.0, label=label)
                        for s, label in SCHEDULE_STYLE.values()],
               loc="lower center", bbox_to_anchor=(0.55, 0.0), ncol=2, fontsize=7.4, handlelength=2.0)
    return fig


# ---------- Derive: physical recruitment ------------------------------------------
def physical_recruitment():
    treatments = json.loads((DYNAMICS.parent / "physical_recruitment.json").read_bytes())["treatments"]
    fig = plt.figure(figsize=(FULL_MM / 25.4, 1.9))
    ax  = fig.add_axes([0.075, 0.22, 0.385, 0.62])
    bx  = fig.add_axes([0.59, 0.22, 0.385, 0.62])
    fig.text(0.012, 0.93, "a", fontsize=10, weight="bold")
    fig.text(0.075, 0.93, "Allocated increment and marginal contribution", fontsize=9, weight="bold")
    fig.text(0.53, 0.93, "b", fontsize=10, weight="bold")
    fig.text(0.59, 0.93, "Full-capture threshold", fontsize=9, weight="bold")
    for target, color, marker in [("PV2", TEAL, "o"), ("ESS", "#AE611A", "s")]:
        rows  = treatments[target]
        power = np.array([row[0] for row in rows])
        for j, symbol, style in [(1, "F", "-"), (2, "s", "--")]:
            center, low, high = np.asarray([row[j] for row in rows]).T
            y     = 100 * (center / center[-1] - 1)
            lower = 100 * (low / high[-1] - 1)
            upper = 100 * (high / low[-1] - 1)
            lower[-1] = upper[-1] = y[-1] = 0.0    # an input divided by itself has no spread
            ax.errorbar(power, y, yerr=[y - lower, upper - y], color=color, linestyle=style, marker=marker,
                        ms=3, lw=1.1, capsize=2, elinewidth=0.6, label=f"{target} {symbol}")
        y, low, high = np.array([row[3] for row in rows]).T
        bx.errorbar(power, y, yerr=[y - low, high - y], color=color, marker=marker, ms=3, lw=1.2,
                    capsize=2, elinewidth=0.6, label=target)
    for panel in (ax, bx):
        panel.set_xticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"])
        panel.set_xlabel("Current ESS power fraction", labelpad=3)
        panel.set_xlim(0.22, 1.035)
    ax.axhline(0, lw=0.55, color="#A7ADB3", zorder=0)
    ax.set(ylabel="Relative change (%)", ylim=(-14, 6), yticks=[-10, -5, 0, 5])
    ax.legend(loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.5)
    bx.axhline(0.5, color="#8C9399", lw=0.7, linestyle=":", zorder=0)
    bx.set(ylabel="Largest capture ratio", ylim=(0.414, 0.540), yticks=[0.44, 0.48, 0.52])
    bx.legend(loc="lower right", ncol=2, columnspacing=0.8, handlelength=1.4)
    return fig


# ---------- Assemble output ------------------------------------------------------
parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--dynamics-only", action="store_true", help="redraw only the monetary outcomes in Figs/dynamics.pdf")
options = parser.parse_args()
targets = {PAPER / "Figs" / "expiry_subsets.pdf": expiry_subsets,
           PAPER / "Figs" / "dynamics.pdf": dynamics,
           PAPER / "Figs" / "physical_recruitment.pdf": physical_recruitment}
for path, draw in targets.items():
    if options.dynamics_only and draw is not dynamics:
        continue
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = draw()
    figure.savefig(path)
    if draw is physical_recruitment:
        figure.savefig(PAPER / "Supplement/figs/physical_recruitment.pdf")
        figure.savefig(PAPER / "Supplement/figs/physical_recruitment.png", dpi=350)
    plt.close(figure)
    print("wrote", path)
