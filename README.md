# Competing VPPs — code, data and figures

Research materials for *Profit-Taking Strategies of Competing Virtual Power Plants Under Time-Variant Switching Costs: Equilibrium Analysis of a Multi-Leader–Multi-Follower Game*.

Both case studies start from electricity-market scenarios and resource data. Python and COPT solve each coalition’s day-ahead and real-time market participation problem. The game uses the resulting coalition profits, the stated allocation rules and the contract terms. With amounts in CNY, the two-VPP, five-DER market has a minimum maximum unilateral gain of **¥160.84** per contract period. The four-VPP, twenty-DER market has a minimum of **¥97.25**. The positive lower bounds exclude a pure equilibrium in both baseline games.

## Check the saved results

Run these commands from the package root. The five-DER certificate uses Python’s standard library. The twenty-DER replay additionally requires NumPy and SciPy.

~~~sh
python verify_release.py
python alternative/five/triangles.py --verify
python alternative/search/audit.py
~~~

The two certificates enclose the minima in [160.8376065314604, 160.8376065314608] and [97.2496112941, 97.2496376192] CNY. Separate COPT traces and full-domain deviation checks are included for each case.

## Reproduce the figures

The numerical panels can be redrawn from their saved full-precision data. The manuscript figures use Times New Roman and the versions in `requirements-figures.txt` (Python 3.10).

~~~sh
python -m pip install -r requirements-figures.txt
python figure_code/render_case_figures.py
~~~

| Manuscript item | Source |
| --- | --- |
| Market timeline and analytic cycle (Figs. 1–2) | `Figs/source/timeline.svg`, `Figs/TimeLine_V2.pdf`, `timeline.py` and `cycle.tex` |
| Five-DER settings and equilibrium (Table III and Fig. 3) | `physical/five_tight.csv`, `physical/five_end_to_end.json` and `paper_rebuilt/five/rebuild_results.py` |
| Twenty-DER contracts (Table IV) and the full-domain upper and lower regret bounds before and after subdivision (Fig. 4) | `physical/inputs/twenty_contracts.mat`, `physical/large/results.json`, `alternative/tightened/market/` and `paper_rebuilt/twenty/global_convergence.py` |
| Common ratios (Fig. 5) | `figure_code/baseline_data.py` and `Supplement/results/fixed_quotes.json` |
| Sensitivity analysis (Fig. 6) | `paper_rebuilt/sensitivity/` |
| Supplementary material | `Supplement/supplement.tex`, `Supplement/code/`, `Supplement/results/` and `Supplement/tables/` |

[REPRODUCE.md](REPRODUCE.md) gives commands for the electricity-market models, game solvers, sensitivity studies and figures. Solver reruns require the packages in `requirements.txt` (Python 3.10) and a working COPT license.

## Data and license

The source electricity-market inputs and attribution are in [physical/source](physical/source/README.md). The underlying electricity-market model is [Open_MLMF_Code](https://github.com/SCWang01/Open_MLMF_Code). The original project code is released under the [MIT license](LICENSE). The [upstream notice](licenses/Open_MLMF_Code-MIT.txt) accompanies the source material. COPT and the manuscript fonts are installed separately. The currency display uses a shared [currency setting](currency.json).
