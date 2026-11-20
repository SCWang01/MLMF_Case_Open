# Competing virtual power plants: code, data and figures

Research materials for *Profit-Taking Strategies of Competing Virtual Power Plants Under Time-Variant Switching Costs: Equilibrium Analysis of a Multi-Leader–Multi-Follower Game*.

The examples ask whether any set of posted profit-taking ratios leaves every operator without a profitable unilateral deviation. The two baseline markets have 2 VPPs and 5 DERs, and 4 VPPs and 20 DERs. The supplied full-domain checks put the smallest achievable maximum gain at **¥160.84** and **¥98.41** per contract period, respectively. Either positive value excludes a pure equilibrium. The package includes the inputs, Python/COPT optimization models, full-precision results, independent checks and figure sources.

## Start with the saved results

Extract the ZIP and run these commands from the directory containing this README. With Python 3.10 or later, the commands below verify file integrity, check the recorded cases and independently verify the five-DER bounds for both sets of dispatch coefficients. They use the Python standard library.

~~~
python verify_release.py
python verify_records.py
python alternative/five/triangles.py --verify
~~~

For the four-VPP certificate, install the scientific dependencies below and run:

~~~
python alternative/search/audit.py
~~~

The four-VPP check should enclose the baseline value between 98.40621 and 98.40623 when the endpoints are rounded outwards to five decimal places. The study uses the unrounded data for all equilibrium tests.

## Recompute the optimization results

The solver runs used Python 3.10.17 and COPT 8.0.6. Use a Python environment with a working COPT license for new solves.

~~~
python -m pip install -r requirements.txt
python small/run_five.py
~~~

The five-DER run first excludes a pure equilibrium in CCG round 3 in the saved trace. Instructions for the larger run, the 14,288 sensitivity configurations, upstream coalition dispatch, and independent certificate checks are in [REPRODUCE.md](REPRODUCE.md). Some commands regenerate result files in place. Use a fresh extraction to compare a rerun with the released records.

## Recreate the figures and tables

The numerical figures can be redrawn from the saved records without a COPT solve. For the published figure layout, use a separate Python 3.12 environment with the plotting dependencies and Times New Roman installed.

~~~
python -m pip install -r requirements-figures.txt
python extract_figure_data.py --case-root .
python figure_code/baseline_data.py
python render_case_figures.py --output-dir Figs
~~~

To use an openly available font, add `--font "DejaVu Serif"` to the last command. This changes the typography of the figures while retaining their plotted data.

| Manuscript item | Source or command | Content |
| --- | --- | --- |
| Fig. 1 | [published vector](Figs/TimeLine_V2.pdf), [editable SVG](Figs/source/timeline.svg), `python timeline.py` | Contract and electricity-market timeline. The Python illustration is a separate redraw and writes `Figs/timeline_python.pdf`. |
| Fig. 2 | `pdflatex -output-directory=Figs cycle.tex` | Exact best responses and the cycle after contract expiry |
| Fig. 3 | `small/results.json` | Five-DER contract expiry and equilibrium bounds |
| Fig. 4 | `large/` and `alternative/` | Four-VPP CCG results and independent global lower bounds |
| Fig. 5 | `figure_code/baseline_data.py` | Predetermined ratios compared with the CCG solution |
| Fig. 6 | `sensitivity/` | Equilibrium classifications and remaining contract terms |
| Tables III and IV | `small/inputs/baseline.json`, `large/data/twenty_case.mat` and `case_setting.json` | Market parameters, membership and contracts |
| Supplement | `python Supplement/code/supplement_figures.py` and `python Supplement/code/supplement_tables.py .` | Supplementary figures and the LaTeX table inputs |

The optional `case_setting.py` draws a portfolio overview from an earlier layout. The paper presents these inputs in Tables III and IV.

Use `latexmk -pdf supplement.tex` from the `Supplement/` directory to compile the supplement. LaTeX is also needed for the analytic figure. The default reporting currency is CNY. See [REPRODUCE.md](REPRODUCE.md#currency-labels) for the shared currency setting.

## Data and code

| Directory | Contents |
| --- | --- |
| [Original physical data](physical/source/README.md) | Five-DER MATLAB inputs and coalition workbook, twenty-DER scenario, demand and coalition workbooks, preprocessing and source hashes |
| `physical/`, `physical_full/` | Market scenarios and resource parameters, two-stage dispatch, allocation calculations, and results for 31 and 2,043 coalitions |
| `small/`, `large/` | Baseline games, COPT CCG code, feasible ratios, iteration traces and subdivision results |
| `alternative/`, `verification/` | Independent full-domain bounds and best-response checks |
| `sensitivity/`, `ancillary/` | Sensitivity configurations, classifications, control cases and exact certificates |
| `figure_code/`, `Figs/`, `Supplement/` | Plotting programs, vector figures, supplementary analyses, tables and their source data |

The original market inputs, sensitivity design and five-DER geometry proofs can also be regenerated independently:

~~~
python physical/prepare_inputs.py --output rebuilt_inputs
python sensitivity/build_design.py --output design_rebuilt.json
python ancillary/generate_certificates.py
~~~

The [sensitivity design notes](sensitivity/design_inputs/README.md) describe how the ten original perturbation tables were obtained. An optional historical generator uses Gurobi. The supplied tables, reconstruction of the sensitivity design and all COPT equilibrium runs can be used without Gurobi.

Use the solver environment above for the published numerical precision. The geometry generator writes 16 certificates under `ancillary/regenerated_certificates/`. The supplied `ancillary/certificates/` are the reference copies. The physical source manifest records how the original files correspond to the solver inputs.

The frozen baseline coefficients reproduce the main figures. The separately recomputed dispatch coefficients give about ¥160.84 and ¥97.25 per contract period. Source data retain full precision. All file paths in `SHA256SUMS.txt` are relative to this directory.

## License

The original source code in this repository is licensed under the [MIT License](LICENSE). The [upstream MIT notice](licenses/Open_MLMF_Code-MIT.txt) accompanies code derived from Open_MLMF_Code.

For dataset provenance and attribution, see the [physical input sources](physical/source/README.md) and [sensitivity-design inputs](sensitivity/design_inputs/README.md). The code license does not grant additional rights to third-party datasets.

## Source attribution

The upstream electricity-market model is [Open_MLMF_Code](https://github.com/SCWang01/Open_MLMF_Code). Its [MIT copyright notice](licenses/Open_MLMF_Code-MIT.txt) accompanies the upstream material. The [source manifest](physical/source/manifest.json) identifies the physical files used in the present study. COPT and the fonts used for the manuscript figures are installed separately.
