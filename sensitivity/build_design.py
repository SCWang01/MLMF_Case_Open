"""Rebuild the three paper sensitivity designs from coalition values and exact quote witnesses."""
import argparse
from functools import lru_cache
from itertools import product
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "design_inputs"
sys.path.insert(0, str(HERE.parent / "Supplement/code"))
from market_family import D, NU, coefficients, members


RULES = ("equal_dp", "shapley")
MIN_SLOPE = 1.0
PAPER = {"masks": (13, 18), "rules": ("equal_dp", "shapley"), "lengths": (5, 4)}


@lru_cache(maxsize=None)
def values_for(data):
    if data is None:
        return NU
    record = json.loads((INPUTS / f"data_{data}.json").read_bytes())
    return {int(k): v for k, v in record["values"].items()}


def markets():
    """The 40 markets: WT sits in VPP 1 so that each unordered split appears once; splits with a
    single-member coalition are left out because that member's own increment is zero."""
    for m1 in range(1, (1 << D) - 1):
        m2 = (1 << D) - 1 ^ m1
        if not m1 & 1 or min(len(members(m1)), len(members(m2))) < 2:
            continue
        for rules in product(RULES, repeat=2):
            yield (m1, m2), rules


def expiry_keys():
    """40 markets x 32 expiry states of the archived data."""
    for masks, rules in markets():
        for free in range(1 << D):
            yield {"family": "expiry", "masks": list(masks), "rules": list(rules), "free": free}


def resim_sets() -> list[int]:
    return sorted(int(path.stem.split("_")[1]) for path in INPUTS.glob("data_*.json"))


def resim_keys():
    """The expiry design on every re-simulated data set. A market enters only if all its allocated
    increments reach MIN_SLOPE (both classifiers need positive increments); skipped markets are
    counted in the summary."""
    for data in resim_sets():
        nu = values_for(data)
        for masks, rules in markets():
            if coefficients(masks, rules, nu)[0].min() < MIN_SLOPE:
                continue
            for free in range(1 << D):
                yield {"family": "resim", "data": data, "masks": list(masks), "rules": list(rules), "free": free}


def age_keys():
    """Every remaining-period vector of the manuscript's market; 0 means an expired contract."""
    incumbent = [0 if PAPER["masks"][0] >> d & 1 else 1 for d in range(D)]
    for ages in product(*[range(PAPER["lengths"][incumbent[d]]) for d in range(D)]):
        yield {"family": "ages", "masks": list(PAPER["masks"]), "rules": list(PAPER["rules"]), "ages": list(ages)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "design_rebuilt.json")
    dest = parser.parse_args().output
    records = json.loads((INPUTS / "witnesses.json").read_bytes())
    witnesses = {json.dumps(r["key"], sort_keys=True): r["witnesses"] for r in records}
    near_zero = json.loads((HERE.parent / "ancillary/near_zero_cases.json").read_bytes())
    certificate_ids = {json.dumps(r["key"], sort_keys=True): r["id"] for r in near_zero}
    all_markets, rows = {}, []
    for family in (expiry_keys, age_keys, resim_keys):
        for key in family():
            masks, rules = key["masks"], key["rules"]
            data = key.get("data")
            name = ",".join(map(str, (*masks, *rules, "base" if data is None else data)))
            if name not in all_markets:
                offers, margins, base = coefficients(masks, rules, values_for(data))
                all_markets[name] = {"f": offers.tolist(), "contribution": margins.T.tolist(),
                                     "base": base.tolist(), "incumbent": [0 if masks[0] >> d & 1 else 1 for d in range(D)],
                                     "epsilon": 0.1}
            row = {"market": name, "key": key}
            canonical = json.dumps(key, sort_keys=True)
            certificate = certificate_ids.pop(canonical, None)
            if certificate is not None:
                row["certificate"] = certificate
            witness = witnesses.pop(canonical, None)
            if witness:
                row["witnesses"] = witness
            rows.append(row)
    if witnesses or certificate_ids:
        raise ValueError("Input witnesses have configuration keys outside the design")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="\n") as out:
        json.dump({"markets": all_markets, "configurations": rows}, out, separators=(",", ":"))
    print(f"{len(all_markets)} markets; {len(rows)} configurations written to {dest}")


if __name__ == "__main__":
    main()
