# Physical inputs

The five-DER MAT file contains the original dispatch parameter structure, including scenario probabilities, prices, production and demand profiles, and equipment constraints. The accompanying workbook contains the reference coalition values.

For twenty DERs, the dispatch export reconstructs each extreme scenario using the historical series' minimum and range. The sampled and normalized scenarios, historical data and reference coalition workbook are under `twenty/`. The three MATLAB files document the original scenario transformation, resource parameters and dispatch formulation. The executable paper reproduction uses `physical/dispatch.py` and the COPT solver.

From the package root, run `python physical/prepare_inputs.py --output rebuilt_inputs`. It creates `five.npz`, `twenty.npz`, `reference.json` and `source_sha256.json`. Omitting `--output` writes to `physical/inputs/`. `manifest.json` maps each archived source path to its filename here and SHA-256 digest.

The twenty-DER coalition workbook and Creating.m match the corresponding files in [Open_MLMF_Code](https://github.com/SCWang01/Open_MLMF_Code). The accompanying upstream notice is [available here](../../licenses/Open_MLMF_Code-MIT.txt).
