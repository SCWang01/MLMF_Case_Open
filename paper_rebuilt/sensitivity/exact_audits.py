"""Verify exact pure-equilibrium claims for all COPT quotes accepted within tolerance."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from fractions import Fraction
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import sys

from exact_witness import ROOT, exact_check, FrozenGame, witness
from sweep import canonical, games, inputs_for

import os
CODE = Path(__file__).resolve().parent
HERE = Path(os.environ.get("MLMF_SENSITIVITY_OUTPUT", CODE)).resolve()
HERE.mkdir(exist_ok=True, parents=True)
SAMPLES = ("base", *map(str, range(10)))


def geometry(inputs, key):
    ancillary = ROOT / "ancillary"
    sys.path.insert(0, str(ancillary))
    import generate_certificates as generate
    import verify_cases as verify

    with redirect_stdout(io.StringIO()):
        report = generate.audit(generate.Game(inputs))
    if not report["pure_equilibrium_excluded"]:
        candidates = [p["point"]["rational"] for p in report["points"] if not p["excluded"]]
        for segment in report["segments"]:
            candidates.extend(p["point"]["rational"] for p in segment["failures"] if p["exact_equilibrium"])
        exact = FrozenGame(inputs)
        for candidate in candidates:
            if witness(exact, tuple(map(Fraction, candidate))):
                return {"witness": candidate, "method": "exhaustive_candidate_geometry"}
        raise ArithmeticError(f"Geometry left unresolved portions for {key}.")
    check = verify.verify(verify.FrozenGame(inputs), report)
    case_id = hashlib.sha256(canonical(key).encode()).hexdigest()[:20]
    destination = HERE / "exact_audits" / "certificates" / f"{case_id}.json.gz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(key=key, inputs=inputs, report=report), separators=(",", ":")).encode()
    destination.write_bytes(gzip.compress(payload, mtime=0))
    return {"witness": None, "exact_excluded": True, "certificate": destination.relative_to(HERE).as_posix(),
            "independent_verification": check}


def check(task):
    key, market, quote = task
    inputs = inputs_for(key, market)
    profile = exact_check(inputs, quote)
    result = {"witness": profile, "method": "exact_global_best_responses"} if profile else geometry(inputs, key)
    return {"key": key, **result}


def read_rows(path):
    if not path.exists():
        return []
    # The writer may still be flushing a trailing partial row.
    with path.open("rb") as stream:
        return [json.loads(line) for line in stream if line.endswith(b"\n")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES), choices=SAMPLES)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--follow", action="store_true", help="Wait for every requested COPT sweep to finish.")
    args = parser.parse_args()
    target = HERE / "exact_audits"
    target.mkdir(exist_ok=True)
    checked = {s: {canonical(r["key"]) for r in read_rows(target / f"{s}.jsonl")} for s in args.samples}
    markets = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while True:
            all_complete = True
            had_work = False
            for sample in args.samples:
                raw = read_rows(HERE / "records" / f"{sample}.jsonl")
                overrides = read_rows(HERE / "repairs" / f"{sample}.jsonl")
                # A repaired result supersedes the original CCG classification for that key.
                latest = {canonical(r["key"]): r for r in (*raw, *overrides)}
                records = list(latest.values())
                if not records:
                    all_complete = False
                    continue
                if sample not in markets:
                    markets[sample] = {canonical(k): m for k, m in games(sample)}
                table = markets[sample]
                keys = {canonical(r["key"]) for r in records}
                if not keys.issubset(table):
                    raise ValueError(f"Unexpected configuration key in sample {sample}.")
                all_complete = all_complete and len(keys) == len(table)
                within = {canonical(r["key"]): r for r in records if r["decision"] == "within"}
                tasks = [(r["key"], table[k], r["quote"]) for k,r in within.items() if k not in checked[sample]]
                if not tasks:
                    continue
                had_work = True
                print(sample, "verifying", len(tasks), "accepted quotes", flush=True)
                with (target / f"{sample}.jsonl").open("a", encoding="utf8") as stream:
                    for row in pool.map(check, tasks, chunksize=1):
                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                        stream.flush()
                        checked[sample].add(canonical(row["key"]))
            if all_complete or not args.follow:
                break
            if not had_work:
                time.sleep(10)


if __name__ == "__main__":
    main()

