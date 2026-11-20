"""Independently replay both certificates and audit the reported feasible quotes."""
from decimal import Decimal, localcontext
from fractions import Fraction
import gzip
import hashlib
import json

from tight_cover import HERE, ROOT, read_game, verify
from check_four_quote import player_optimum


def decimal4(value):
    with localcontext() as context:
        context.prec=80
        return format(Decimal(value.numerator)/Decimal(value.denominator),".4f")


def main():
    results={}
    for case,target,relative in (("frozen","98.40621","large/data/twenty_case.mat"),
                                 ("dispatch","97.24961","physical/twenty_end_to_end.mat")):
        src=ROOT/relative
        blob=(HERE/f"cover_{case}_{target}.json.gz").read_bytes()
        proof=verify(src,json.loads(gzip.decompress(blob)),Fraction(target))
        search=json.loads((HERE/f"search_{case}.json").read_bytes())
        if search["source_sha256"]!=proof["input_sha256"]:
            raise ValueError("Search and certificate used different inputs")
        game=read_game(src)
        quote=tuple(map(Fraction,search["best"]["quote"]))
        if len(quote)!=4 or any(q<0 or q>1 for q in quote):
            raise ValueError("Infeasible candidate")
        regret=[player_optimum(game,quote,v)["regret"] for v in range(4)]
        upper=max(regret)
        lower=Fraction(proof["exact_lower"])
        if upper!=Fraction(search["best"]["exact_upper"]) or not 0<=lower<=upper:
            raise ArithmeticError("Global bound or candidate audit failed")
        if decimal4(lower)!=decimal4(upper):
            raise ArithmeticError("Four-decimal rounding is not uniquely determined")
        results[case]=dict(lower=float(lower),upper=float(upper),gap=float(upper-lower),
                           global_minimax_rounded_4dp=decimal4(lower),
                           checked_quote=list(map(float,quote)),regrets=list(map(float,regret)),
                           exact_lower=str(lower),exact_upper=str(upper),integer_checked_leaves=proof["leaves"],
                           source_sha256=proof["input_sha256"],certificate_sha256=hashlib.sha256(blob).hexdigest())
    # The interval for the difference propagates both independently verified brackets.
    f,d=results["frozen"],results["dispatch"]
    diff=(Fraction(f["exact_lower"])-Fraction(d["exact_upper"]),
          Fraction(f["exact_upper"])-Fraction(d["exact_lower"]))
    results["frozen_minus_dispatch"]=[float(x) for x in diff]
    print([(k,v["lower"],v["upper"]) for k,v in (("frozen",f),("dispatch",d))],flush=True)
    (HERE/"comparison.json").write_text(json.dumps(results,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()

