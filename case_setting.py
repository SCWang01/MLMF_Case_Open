"""Render both contract portfolios. Optionally extract their data from the input games."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
TYPES = ["PV1", "PV2", "TG", "ESS", "DR"]
COLORS = ["#c3dcf1", "#deebf5", "#f3dcc2", "#cbe2dc", "#e4dbec"]


def read_contracts(five, twenty):
    from scipy.io import loadmat
    game = json.loads(five.read_bytes())["inputs"]
    lengths = (5, 4)
    small = []
    for operator, rule in enumerate(("EPD", "Shapley")):
        members = [i for i, o in enumerate(game["incumbent"]) if o == operator]
        fees = {game["compensation"][operator][i] for i in members}
        assert len(fees) == 1
        fee = fees.pop()
        contracts = []
        for i in members:
            rival = 1 - operator
            term = game["c"][i][rival] * lengths[rival] / fee
            assert abs(term - round(term)) < 1e-12
            contracts.append([TYPES[i], TYPES[i], round(term)])
        small.append(dict(rule=rule, length=lengths[operator], gamma=int(fee), members=contracts))
    data = loadmat(twenty, simplify_cells=True)["cfg"]
    types = [t for t, count in zip(TYPES, data["VPP_List"]) for _ in range(count)]
    large = []
    rules = {1: "EPD", 2: "Shapley", 3: "Nucleolus"}
    for vpp in data["vpp"]:
        contracts = [[str(d), types[d - 1], int(r)] for d, r in zip(vpp["members"], vpp["member_remain"])]
        large.append(dict(rule=rules[vpp["allocation_method"]], length=vpp["contract_length"],
                          gamma=vpp["switching_factor"], members=contracts))
    assert sorted(int(c[0]) for row in large for c in row["members"]) == list(range(1, 21))
    return {"five": small, "twenty": large}


def draw(data, destination, review_exports):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7.3, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "text.color": "#172735"})
    fig, ax = plt.subplots(figsize=(3.48, 2.90))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")

    def label(x, y, value, **kwargs):
        ax.text(x, y, value, va="center", **kwargs)

    groups = (("five", "(a)  Two VPPs · five DERs", 95),
              ("twenty", "(b)  Four VPPs · twenty DERs", 59))
    for key, title, top in groups:
        label(1, top, title, fontsize=9, weight="bold")
        y_header = top - 6
        for x, title, alignment in [(1, "VPP / rule", "left"), (30, "L", "center"),
                                    (41, "γ", "center"), (50, "DER / remaining periods", "left")]:
            label(x, y_header, title, ha=alignment, fontsize=7.1, color="#415160")
        ax.plot([1, 99], [y_header-3, y_header-3], color="#a7b1b8", lw=.55)
        for operator, row in enumerate(data[key], 1):
            y = y_header - 8.7 - (operator - 1) * 9.7
            label(1, y, f"{operator}  {row['rule']}", fontsize=7.2)
            label(30, y, row["length"], ha="center")
            label(41, y, row["gamma"], ha="center")
            for j, (identity, kind, term) in enumerate(row["members"]):
                left = 50 + j * 8.22
                ax.add_patch(Rectangle((left, y-4.1), 7.7, 8.2, lw=0,
                                       facecolor=COLORS[TYPES.index(kind)]))
                label(left+3.85, y+1.9, identity, ha="center", fontsize=6.65, weight="bold")
                label(left+3.85, y-2.0, term, ha="center", fontsize=6.65)
    for x, kind, color in zip((1, 22, 43, 61, 81), TYPES, COLORS):
        ax.add_patch(Rectangle((x, 0.9), 3.4, 2.9, facecolor=color, edgecolor="#71818a", lw=.4))
        label(x+4.6, 2.35, kind, fontsize=7)
    fig.savefig(destination, format="pdf", metadata={"Creator":"Matplotlib", "CreationDate":None, "ModDate":None})
    if review_exports:
        fig.savefig(destination.with_suffix(".svg"), format="svg")
        fig.savefig(destination.with_suffix(".png"), format="png", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five", type=Path, help="five-DER baseline JSON, together with --twenty")
    parser.add_argument("--twenty", type=Path, help="twenty-DER baseline MAT")
    parser.add_argument("--data", type=Path, default=HERE / "case_setting.json")
    parser.add_argument("--output", type=Path, default=HERE / "case_setting.pdf")
    parser.add_argument("--review-exports", action="store_true")
    args = parser.parse_args()
    if bool(args.five) != bool(args.twenty):
        parser.error("Supply both game inputs to rebuild the figure data.")
    if args.five:
        args.data.write_text(json.dumps(read_contracts(args.five, args.twenty), indent=2) + "\n", encoding="utf-8")
    draw(json.loads(args.data.read_bytes()), args.output, args.review_exports)


if __name__ == "__main__":
    main()