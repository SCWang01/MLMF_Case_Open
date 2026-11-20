# Reproducing the case studies

The manuscript evaluates the switching-window game at the supplied full-precision market coefficients. The two primary cases have two VPPs and five DERs, and four VPPs and twenty DERs. The package also contains a separate reconstruction of their upstream coalition dispatch and allocation models.

Use the solver environment described in [README.md](README.md). Running new optimization problems requires a working COPT license with nonconvex mixed-integer support. Recorded data, the figure renderer and independent certificate checks are available after extraction. Run the commands below from the extracted directory. Time-limited optimization trajectories can vary with hardware and scheduling.

## Reproduce the manuscript figures

```sh
python extract_figure_data.py --case-root .
python figure_code/baseline_data.py
python render_case_figures.py --output-dir Figs
```

The five-DER case presents its VPP allocation rules, contract lengths and switching-cost factors in the text, and Table III lists each DER's initial VPP, remaining term and payoff coefficients. In the twenty-DER case, Table IV lists the initial contracts and allocation rules for all four VPPs. The contract settings for both cases are recorded in `case_setting.json`. Figures 3 and 4 show the five-DER and twenty-DER equilibrium checks. Figure 5 compares common posted ratios with the minimum-regret profiles. Figure 6 shows the sensitivity results. `figure_data.json` identifies the source records by SHA-256. The manuscript renderer defaults to Times New Roman. Pass `--font "DejaVu Serif"` for a portable alternative. The released figures 3–6 were rendered with Matplotlib 3.9.2. The figure source map and the separate commands for Figures 1 and 2 are in [README.md](README.md).

## Currency labels

The released figures and supplementary tables report Chinese yuan (CNY, ¥). From the extracted directory, select the currency label and rebuild the figures and supplementary PDF with one of these commands.

```sh
python set_currency.py CNY
python set_currency.py USD
python set_currency.py SGD
```

For another currency, edit the `code`, `name`, and `symbol` fields in `currency.json`, then run `python set_currency.py`. The command generates `currency.tex`, redraws the figures, and compiles the supplementary PDF. Changing the label leaves the numerical amounts in the units supplied by the case. An exchange-rate conversion requires rescaling the monetary inputs and results separately.

## Compute the primary results

```sh
python small/run_five.py
python large/run.py --workers 7 --presolve 0
python alternative/refine.py --case frozen --workers 9 --nodes 81 --seconds 3 --budget 120
python sensitivity/run_copt.py --all --workers 4
python sensitivity/run.py
python sensitivity/mixed_bounds.py
python sensitivity/retry.py
python sensitivity/summarize.py
python ancillary/sweep.py --workers 6
```

- `small/results.json` contains both the numerical decision run and the continued refinement. The decision run excludes in round 3. The refinement produces 160.84 per contract period in the selected monetary unit to two decimals.
- `large/runs/` contains the full-domain and local COPT CCG master traces. With the recorded budgets, the full-domain run stops after eight rounds with a checked upper bound of 123.130679. The local CCG runs exclude 10 of 23 roots and leave a global lower bound of zero.
- `alternative/refinement/frozen.json` contains all 23 regional results of the additional COPT MILP relaxation and bisection. Their minimum lower bound is 0.139810594, above 0.001001. Of the 23 original node logs, 19 were retained. The four other roots have original summary bounds plus fresh detailed COPT runs under `alternative/replayed_roots/`. Combining the 19 original and four replayed trees reproduces the same global minimum. Run `python alternative/complete_traces.py` to repeat those four solves.
- The tighter four-VPP interval [98.4062116831, 98.4062299248] is supported by the independent integer subdivision certificate and full-domain rational best-response check.  `python alternative/search/search.py --case frozen` produces feasible quotes. `python alternative/search/tight_cover.py --case frozen --generate` generates the lower-bound certificate.
- `sensitivity/copt_full.csv` contains first-pass results for all 14,288 cases. `copt_retries.json` records 47 presolve-off retries and `copt_audited.csv` contains the accepted COPT classifications. The three configuration families have 182/1,280, 880/2,000 and 1,647/11,008 cases within tolerance. Input design keys, previously recorded feasible witnesses and the historical comparison column are supplied for regression checks. The published counts in this revision are counted from the COPT decision field. Rational best responses and whole-square deviation bounds give an independent cross-check on every decision.
- `ancillary/results.json` contains 69 new COPT runs. These cover the positive control, four baseline smoothing settings, and 32 expiry masks at each of two smoothing settings. The two groups have 11 and 12 independently confirmed exact equilibrium witnesses. At the stored COPT positive-control quote, the independently checked maximum regret is 1.95060e-17.

The supplementary figures, tables, simulations, fixed-quote checks, physical recruitment study and large-smoothing runs have their own sources and inputs under `Supplement/`. Use the following commands from the project root to regenerate figures and tables from the saved outputs.

~~~sh
python Supplement/code/supplement_figures.py
python Supplement/code/supplement_tables.py .
~~~

The supplementary PDF and LaTeX input list how the reported results correspond to the figures and tables.

## Replay certificates and check saved runs

```sh
python verify_records.py
python alternative/check_recorded_cover.py
python alternative/five/triangles.py --verify
python alternative/search/audit.py
python ancillary/verify_geometry.py
python sensitivity/summarize.py
```

The two-operator triangle checker verifies both sets of coefficients using exact rational arithmetic. The four-operator checker reconstructs each complete dyadic cover and checks feasible deviations using integer bounds. It separately checks the saved feasible quotes against all unilateral deviations. The replay covers the frozen and rebuilt cases. The sixteen exact five-DER nonexistence certificates in `ancillary/certificates/` concern the frozen baseline and fifteen perturbations with small positive regret. For these fifteen cases, `sensitivity/near_zero_copt.py` records unrounded COPT quotes in a separate rerun. The maximum independently checked regret, 8.4698513092e-7, agrees with the full-sweep COPT records. A positive regret below the numerical tolerance and an exact zero-regret profile are distinct outcomes.

## Rebuild upstream coalition coefficients

```sh
python physical/dispatch.py --case five --all-five --seconds 45 --relative-gap 1e-9 --output physical/five_tight.csv
python physical/rebuild_five.py
python physical_full/run.py --workers 8 --time-limit 35
python physical_full/run.py --workers 8 --time-limit 120 --resume
python physical/rebuild_twenty.py --dispatch physical_full/coalitions.csv
```

`physical/inputs/` contains the two physical inputs and the published reference coalitions. The saved COPT tables cover all 31 nonempty five-DER coalitions and all 2,043 specified twenty-DER coalitions. The largest recorded relative gap for the latter is below 1e-7. With rebuilt coefficients, the minimax regrets are about 160.84 and 97.25 per contract period in the selected monetary unit. The equilibrium tests use unrounded quotes and bounds. The manuscript's case figures use the frozen coefficients. Changing the physical inputs or optimizer settings produces a new case and calls for freshly generated certificates.

`SHA256SUMS.txt` lists the released files. It identifies the bundled snapshot before any reruns regenerate solver outputs or figures.
