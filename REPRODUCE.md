# Reproducing the COPT case studies

Run commands from the extracted package root. Read the checksum manifest with `python verify_release.py` before regenerating files in place. Use Python 3.10 and `python -m pip install -r requirements.txt` with a COPT license for new market-participation, allocation and CCG solves. Recorded runs used COPT 8.0.6. The saved solver outcomes have full precision. Hardware, tolerances and scheduling can affect time-limited trajectories and solver incumbents.

## Start from electricity-market inputs

~~~sh
python physical/prepare_inputs.py
python physical/market_participation.py --case five --all-five --seconds 45 --relative-gap 1e-9 --output physical/five_tight.csv
python physical/rebuild_five.py
python physical_full/run.py --workers 8 --time-limit 35
python physical_full/run.py --workers 8 --time-limit 120 --resume
python physical/rebuild_twenty.py
~~~

The electricity-market inputs are `physical/inputs/five.npz` and `twenty.npz`. The contract files store membership, switching terms and allocation settings. The four-VPP portfolios and their potential entrants determine the set of 2,043 coalitions. For five DERs the model evaluates all 31 nonempty coalitions. Each COPT row records a feasible coalition profit and objective bound. The allocation programs use the feasible profits.

The five-DER program also evaluates the protected contract states, computes the baseline CCG bounds and checks candidate ratios with rational best responses. The twenty-DER output is `physical/twenty_end_to_end.mat`. Its byte fingerprint identifies the game used throughout the four-operator proofs. Re-solving the upstream optimization may change the last digits of an incumbent. For a changed input, regenerate the corresponding equilibrium proofs.

## Equilibrium computations and independent certificates

~~~sh
python large/run.py --workers 7 --presolve 0
python alternative/refine.py --workers 9 --nodes 81 --seconds 3 --budget 120 --output alternative/refinement
python alternative/refine.py --workers 8 --nodes 101 --seconds 4 --budget 130 --presolve 0 --continue-from alternative/refinement --output alternative/tightened
python alternative/five/triangles.py
python alternative/search/search.py
python alternative/search/tight_cover.py --generate
python alternative/five/triangles.py --verify
python alternative/search/audit.py
~~~

For four operators, `physical/large/results.json` and `alternative/tightened/market.json` hold the COPT master bounds on the full domain and 23 subdomains. The final interval uses an independently certified global lower bound and an upper bound computed by checking every unilateral deviation at one feasible profile. The two-dimensional proof substitutes rational deviations on triangles. The four-dimensional proof covers 14,971 leaves using integer bounds. The published certificate files bind the proofs to the saved market inputs.

## Numerical figures, currency and supplement

For Figure 4, `paper_rebuilt/twenty/build.py` replays the four initial CCG rounds and the saved refinements of the 23 subdomains. At sweep zero it includes the verified initial feasible quotes and the first bounds for all subdomains. Each subsequent sweep takes one more completed split from each unfinished subdomain and retains the last cover of each finished subdomain. The plotted lower bound is the minimum across this full cover. The upper bound uses the best feasible quote available by that sweep.

~~~sh
python paper_rebuilt/five/rebuild_results.py
python paper_rebuilt/twenty/build.py
python paper_rebuilt/twenty/prepare_supplement.py Supplement/code/inputs
python Supplement/code/fixed_quotes.py
python extract_figure_data.py
python figure_code/baseline_data.py
python figure_code/render_case_figures.py
~~~

The first two programs recover the tables, deviations and plotted traces from the per-case result records. Use the package’s plotting environment in `requirements-figures.txt` to reproduce the published appearance. The release also includes the editable timeline, the analytic cycle and the compiled figure PDFs.

~~~sh
python ancillary/positive_control.py
python Supplement/code/epsilon_stress.py
python Supplement/code/contract_expiry.py
python Supplement/code/dynamics.py
python Supplement/code/physical_recruitment.py
python Supplement/code/supplement_figures.py
python Supplement/code/supplement_tables.py .
~~~

To build the supplementary PDF, run `latexmk -pdf supplement.tex` in `Supplement/`. The currency configuration is shared across the figures and supplementary material. From the package root run `python set_currency.py CNY`, `python set_currency.py SGD` or `python set_currency.py USD` with the plotting and LaTeX dependencies installed. This changes the currency label; converting numerical amounts requires an exchange rate.

## Sensitivity study

Ten perturbed electricity markets are solved in COPT. Their coalition profits, together with baseline-market profits, feed the membership, allocation-rule and contract-term variants. Each accepted CCG profile is checked against full-domain best responses. The package includes the per-configuration decisions, interval bounds and exact witness or exclusion records.

~~~sh
python paper_rebuilt/sensitivity/market_inputs.py
python paper_rebuilt/sensitivity/sweep.py
python paper_rebuilt/sensitivity/check_bounds.py
python paper_rebuilt/sensitivity/sweep.py --repair
python paper_rebuilt/sensitivity/exact_audits.py
python paper_rebuilt/sensitivity/summarize.py
~~~

The sensitivity commands can resume from their recorded outputs. For a fresh computation, set the environment variable `MLMF_SENSITIVITY_OUTPUT` to a new, empty results directory before running the commands. Each command will read its predecessor’s output from that directory.
