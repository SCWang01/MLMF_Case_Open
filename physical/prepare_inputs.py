"""Rebuild the dispatch inputs from the raw MAT and spreadsheet files in physical/source/."""
import argparse
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.io import loadmat
from xml.etree import ElementTree as ET
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"


def parse_output():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "inputs",
                        help="destination directory (default: physical/inputs)")
    return parser.parse_args().output


def workbook(path):
    # The original sources are each a single-sheet XLSX. Read the cached cells
    # directly so building the numerical inputs needs only NumPy and SciPy.
    tag = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(n.text or "" for n in item.iter(tag + "t")) for item in shared]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.iter(tag + "row"):
        cells = []
        for cell in row.iter(tag + "c"):
            col = 0
            for letter in cell.get("r"):
                if not letter.isalpha():
                    break
                col = 26*col + ord(letter.upper())-ord("A")+1
            cells.extend([None]*(col-len(cells)))
            kind = cell.get("t","n")
            value = cell.findtext(tag + "v")
            if kind == "s":
                value = strings[int(value)]
            elif kind == "inlineStr":
                value = "".join(n.text or "" for n in cell.iter(tag + "t"))
            elif kind == "n" and value is not None:
                value = float(value) if any(mark in value.lower() for mark in (".", "e")) else int(value)
            cells[col-1] = value
        rows.append(cells)
    return rows


def references(path, count):
    rows = workbook(path)[1:]
    return {",".join(str(i + 1) for i, flag in enumerate(row[:count]) if flag): row[count]
            for row in rows}


def save_inputs(path, parameters):
    """Match the released NPZ integer representation across NumPy versions."""
    arrays = {name: np.asarray(value) for name, value in parameters.items()}
    for name, values in arrays.items():
        if np.issubdtype(values.dtype, np.integer):
            encoded = values.astype("<i4")
            if not np.array_equal(encoded, values):
                raise ValueError(f"{name} exceeds the released 32-bit integer format")
            arrays[name] = encoded
    np.savez_compressed(path, **arrays)


def main():
    output = parse_output()
    manifest = json.loads((SOURCE / "manifest.json").read_bytes())
    for item in manifest:
        raw = (SOURCE / item["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Unexpected contents of {item['file']}")
    output.mkdir(exist_ok=True, parents=True)
    five_mat = SOURCE / "five" / "5DER_electricity.mat"
    small_table = SOURCE / "five" / "5DER_coalition_values.xlsx"
    p = loadmat(five_mat, simplify_cells=True)["Parameter"]
    physical = {k: np.atleast_1d(p[k]) for k in
                ("P", "Lmpda", "Lmprt", "pane", "ag", "bg", "cg", "Ug", "Dg", "Pgmin", "Pgmax",
                 "yt_sin", "yt_sout", "as", "socmin", "socmax", "p_cmax", "p_dmax", "Esmax", "Cutting")}
    physical.update(renewable=np.stack([p["Wind_f"], p["Solar_f"]]), dr_rt=p["Load_f"][None],
                    dr_da=p["Load_base"][None], counts=[2, 1, 1, 1])
    save_inputs(output / "five.npz", physical)
    large = SOURCE / "twenty"
    extreme = large / "Data_Scenario_Large_Extreme.xlsx"
    historical = large / "Dataset_Large_Case_Used_50.xlsx"
    names, values = workbook(extreme)
    scenarios = {name: np.asarray(json.loads(entry), dtype=float) for name, entry in zip(names, values)}
    base_rows = workbook(historical)
    history = {name: np.array(column, dtype=float) for name, column in zip(base_rows[0], zip(*base_rows[1:]))}

    def restore(name):
        return scenarios[name] * np.ptp(history[name]) + np.min(history[name])

    columns = [f"{group}_{j + 1}" for group in ("wpp", "pvp") for j in range(5)]
    dr = [f"dr_{i + 1}" for i in range(6)]
    prices = restore("LMPDA") / 4
    large_inputs = dict(counts=[10, 1, 3, 6], P=scenarios["p"].ravel(), Lmpda=prices, Lmprt=prices,
                        renewable=np.stack([restore(k) / 1000 for k in columns]),
                        dr_da=np.stack([history[k] / 1000 for k in dr]),
                        dr_rt=np.stack([restore(k) / 1000 for k in dr]), pane=[500],
                        ag=[350], bg=[0], cg=[300], Pgmin=[0.1], Pgmax=[0.6], Ug=[0.1], Dg=[0.1],
                        yt_sin=[0.95,0.99,0.95], yt_sout=[0.95,0.99,0.95],
                        socmin=[0.05,0.05,0.1], socmax=[0.95,0.95,0.98], Esmax=[8,8,10],
                        p_cmax=[1.5,1.5,1.8], p_dmax=[1.5,1.5,1.8],
                        Cutting=[685,651.45,667.20,700.56,689.86,633.36])
    large_inputs["as"] = np.zeros(3)
    save_inputs(output / "twenty.npz", large_inputs)
    large_table = large / "Possible_Coalition_DER20_VPP4_2026_08_20_11_10_35_464.xlsx"
    refs = {"five":references(small_table, 5), "twenty":references(large_table, 20)}
    (output / "reference.json").write_text(json.dumps(refs, indent=1), encoding="utf-8")
    source_hashes = {item["workspace_path"]: item["sha256"] for item in manifest}
    (output / "source_sha256.json").write_text(json.dumps(source_hashes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
