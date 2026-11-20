# Electricity-market inputs

The five-DER MAT file contains the prices, scenarios and resource parameters. The twenty-DER workbooks contain normalized scenarios and the historical series used to recover the physical scales. The source manifest records the original filenames and SHA-256 digests. The upstream electricity-market model is available at [Open_MLMF_Code](https://github.com/SCWang01/Open_MLMF_Code).

`python physical/prepare_inputs.py` reproduces the two numerical input arrays. The COPT market-participation model and the portfolio definitions determine the coalition values used throughout the study.
