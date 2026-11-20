"""Build contract-game coefficients from COPT coalition market-participation profits."""
import json
import math
from itertools import combinations
from pathlib import Path
import numpy as np
NAMES=("PV1","PV2","TG","ESS","DR")
D=len(NAMES)
NU={int(k):v for k,v in json.loads((Path(__file__).parent/"inputs"/"five_coalitions.json").read_bytes()).items()}
if set(NU) != set(range(32)):
    raise ValueError("Missing coalition values")

def members(mask: int) -> list[int]:
    return [d for d in range(D) if mask >> d & 1]


def label(mask: int) -> str:
    return "+".join(NAMES[d] for d in members(mask)) or "-"


def surplus(mask: int, nu: dict[int, float] = NU) -> float:
    """Cooperative surplus I_S = nu(S) - sum of stand-alone values."""
    return nu[mask] - sum(nu[1 << d] for d in members(mask))


def equal_dp(mask: int, nu: dict[int, float] = NU) -> dict[int, float]:
    """Equal propensity to disrupt: increments proportional to the marginal surplus contributions."""
    marginal = {d: surplus(mask, nu) - surplus(mask ^ (1 << d), nu) for d in members(mask)}
    total    = sum(marginal.values())
    return {d: (surplus(mask, nu) * marginal[d] / total if total else 0.0) for d in marginal}


def shapley(mask: int, nu: dict[int, float] = NU) -> dict[int, float]:
    """Shapley value of the surplus game restricted to the coalition."""
    inside = members(mask)
    n      = len(inside)
    out    = {}
    for d in inside:
        rest  = [j for j in inside if j != d]
        value = 0.0
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for subset in combinations(rest, size):
                sub    = sum(1 << j for j in subset)
                value += weight * (surplus(sub | 1 << d, nu) - surplus(sub, nu))
        out[d] = value
    return out


ALLOCATE = {"equal_dp": equal_dp, "shapley": shapley}


def coefficients(masks, rules, nu: dict[int, float] = NU):
    """Return F (D x 2), s (D x 2), A (2,) of the frozen game for incumbent coalitions `masks`."""
    F = np.zeros((D, 2)); s = np.zeros((D, 2)); A = np.zeros(2)
    for v in range(2):
        S = masks[v]
        for d in range(D):
            T       = S | 1 << d
            F[d, v] = ALLOCATE[rules[v]](T, nu)[d]
            s[d, v] = (surplus(S, nu) - surplus(S ^ (1 << d), nu) if S >> d & 1
                       else surplus(T, nu) - surplus(S, nu))
        A[v] = surplus(S, nu) - sum(s[d, v] for d in members(S))
    return F, s, A


