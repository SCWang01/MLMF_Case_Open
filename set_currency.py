"""Set the currency label used in the manuscript, supplement and figures.

Run python set_currency.py SGD to apply a preset, or edit currency.json and run
python set_currency.py to apply a custom currency. Monetary values are unchanged.
Requires the figure dependencies, Times New Roman and latexmk with pdfLaTeX.
"""
import argparse
from contextlib import contextmanager
from copy import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from shutil import copymode
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory, gettempdir
from zipfile import ZipFile

PRESETS = {
    "CNY": {"code": "CNY", "name": "Chinese yuan", "symbol": "¥"},
    "SGD": {"code": "SGD", "name": "Singapore dollars", "symbol": "S$"},
    "USD": {"code": "USD", "name": "US dollars", "symbol": "$"},
    "EUR": {"code": "EUR", "name": "euros", "symbol": "€"},
    "GBP": {"code": "GBP", "name": "pounds sterling", "symbol": "£"},
}
ARCHIVE = "MLMF_Figure_Code.zip"
MANIFEST = "SHA256SUMS.txt"
FIGURES = ("Figs/case_five_ccg.pdf", "Figs/case_twenty_ccg.pdf",
           "Figs/case_baseline.pdf", "Figs/dynamics.pdf")


def currency_tex(currency):
    code, name, symbol = (currency[k] for k in ("code", "name", "symbol"))
    if not all(isinstance(v, str) for v in (code, name, symbol)):
        raise ValueError("currency.json requires three text fields: code, name, symbol")
    if not re.fullmatch(r"[A-Z]{3}", code):
        raise ValueError("code must contain three uppercase letters")
    if not re.fullmatch(r"[A-Za-z][A-Za-z .()'-]*", name):
        raise ValueError("name must be a plain English currency name")
    if not re.fullmatch(r"(?:[A-Za-z]{1,8}\$?|\$|¥|￥|€|£)", symbol):
        raise ValueError("use a letter label or a currency symbol such as ¥, $, S$, € or £")
    rendering = {"¥": r"\textyen", "￥": r"\textyen", "$": r"\$",
                 "€": r"{\fontfamily{cmr}\selectfont\texteuro}", "£": r"\textsterling"}
    mark = rendering.get(symbol, symbol.replace("$", r"\$"))
    tex = (r"\providecommand{\CurrencyName}{" + name + "}\n"
           + r"\providecommand{\CurrencyCode}{" + code + "}\n"
           + r"\providecommand{\CurrencySymbol}{\ifmmode\text{" + mark
           + r"}\else " + mark + r"\fi}" + "\n")
    return tex.encode("ascii")


def sha(data):
    return sha256(data).hexdigest()


def manifest_bytes(hashes):
    return "".join(f"{hashes[n]}  {n}\n" for n in sorted(hashes)).encode("utf-8")


def update_archive(original, output, changes):
    with ZipFile(original) as src, ZipFile(output, "w") as dst:
        hashes = {}
        for info in src.infolist():
            n = info.filename
            if n == MANIFEST:
                continue
            data = changes[n] if n in changes else src.read(info)
            dst.writestr(copy(info), data)
            if not info.is_dir():
                hashes[n] = sha(data)
        dst.writestr(MANIFEST, manifest_bytes(hashes))
        dst.comment = src.comment
    with ZipFile(output) as z:
        if z.testzip() is not None:
            raise ValueError("updated reproduction archive failed its integrity check")


@contextmanager
def exclusive_build(root):
    # The operating system releases the lock even if the process is interrupted.
    key = sha(os.path.normcase(str(root)).encode("utf-8"))
    with (Path(gettempdir()) / ("mlmf-currency-" + key + ".lock")).open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("another currency update is running in this folder") from exc
        yield


def write_atomic(path, data):
    with NamedTemporaryFile(prefix=".currency-", suffix=".tmp", dir=path.parent,
                            delete=False) as stream:
        tmp = Path(stream.name)
    try:
        tmp.write_bytes(data)
        if path.is_file():
            copymode(path, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def build(root, stage, target):
    inputs = {}

    def read(n):
        data = (root / n).read_bytes()
        inputs[n] = sha(data)
        return data

    def copy_input(n):
        dest = stage / n
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(read(n))

    currency = PRESETS[target] if target else json.loads(read("currency.json").decode("utf-8-sig"))
    tex = currency_tex(currency)
    currency = {k: currency[k] for k in ("code", "name", "symbol")}
    # The English text font includes the ordinary yen/yuan sign.
    if currency["symbol"] == "￥":
        currency["symbol"] = "¥"
    config = (json.dumps(currency, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    (stage / "currency.json").write_bytes(config)
    (stage / "currency.tex").write_bytes(tex)
    has_main = (root / "main.tex").is_file()
    for n in ("figure_code/render_case_figures.py", "figure_code/figure_data.json",
              "figure_code/baseline_data.json", "Supplement/code/supplement_figures.py",
              "Supplement/results/dynamics.json", "Supplement/supplement.tex"):
        copy_input(n)
    for directory, mask in [("Supplement/tables", "*.tex"), ("Supplement/figs", "*.pdf")]:
        for p in (root / directory).glob(mask):
            copy_input(p.relative_to(root).as_posix())
    if has_main:
        for n in ("main.tex", "latexmkrc", "main.bbl"):
            if (root / n).is_file():
                copy_input(n)
        for mask in ("*.cls", "*.bst", "*.bib"):
            for p in root.glob(mask):
                copy_input(p.name)
        main_source = (stage / "main.tex").read_text(encoding="utf-8-sig")
        active = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{(Figs/[^}]+)\}", main_source)
        for n in dict.fromkeys(active):
            if n not in FIGURES:
                copy_input(n)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")

    def run(args, cwd):
        subprocess.run(args, cwd=cwd, env=env, check=True)

    run([sys.executable, "-B", "figure_code/render_case_figures.py"], stage)
    # The percentage-only sensitivity figure uses the manuscript's existing artwork.
    if has_main and "Figs/case_sensitivity.pdf" in active:
        copy_input("Figs/case_sensitivity.pdf")
    run([sys.executable, "-B", "Supplement/code/supplement_figures.py", "--dynamics-only"], stage)
    latex = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error"]
    run(latex + ["supplement.tex"], stage / "Supplement")
    outputs = ["currency.json", "currency.tex", *FIGURES, "Supplement/supplement.pdf"]
    if has_main:
        run(latex + ["main.tex"], stage)
        outputs.append("main.pdf")
        if (stage / "main.synctex").is_file():
            sync = stage / "main.synctex"
            sync.write_bytes(sync.read_bytes().replace(stage.as_posix().encode(),
                                                       root.as_posix().encode()))
            outputs.append("main.synctex")
    changes = {n: (stage / n).read_bytes() for n in outputs}
    if (root / MANIFEST).is_file():
        hashes = {}
        for line in read(MANIFEST).decode("utf-8-sig").splitlines():
            h, n = line.split("  ", 1)
            hashes[n] = sha(changes[n]) if n in changes else h
        changes[MANIFEST] = manifest_bytes(hashes)
    if (root / ARCHIVE).is_file():
        read(ARCHIVE)
        update_archive(root / ARCHIVE, stage / ARCHIVE, changes)
        changes[ARCHIVE] = (stage / ARCHIVE).read_bytes()
    for n, h in inputs.items():
        if sha((root / n).read_bytes()) != h:
            raise RuntimeError(f"source changed during the build: {n}")
    previous = {n: (root / n).read_bytes() if (root / n).exists() else None for n in changes}
    installed = []
    try:
        for n, data in changes.items():
            if data != previous[n]:
                write_atomic(root / n, data)
                installed.append(n)
    except BaseException:
        for n in reversed(installed):
            if previous[n] is None:
                (root / n).unlink()
            else:
                write_atomic(root / n, previous[n])
        raise
    return currency["code"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("code", nargs="?", type=str.upper, choices=sorted(PRESETS),
                        help="omit to apply the existing currency.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with exclusive_build(root), TemporaryDirectory(prefix="mlmf-paper-currency-") as temp:
        code = build(root, Path(temp), args.code)
    print(f"Applied {code} to monetary labels in the paper files.")


if __name__ == "__main__":
    main()

