"""Audit one minimax candidate with exact rational arithmetic, without Gurobi.

Every saved binary64 input is interpreted as its exact rational value. DER
projection, original signed MATLAB profits and all unilateral maxima are then
computed using Fraction. This certifies only the candidate's regret upper
bound; it never certifies the solver's global minimax lower bound.

Run: python -B equilibrium_research/audit_minimax_4vpp_candidate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "physical/twenty_end_to_end.mat"
DEFAULT_CANDIDATE = ROOT / "runs" / "full_1.json"
DEFAULT_OUTPUT = ROOT / "full_candidate_audit.json"
ZERO = Fraction(0)
ONE = Fraction(1)


# Convert a finite stored binary64 value into its exact rational representation.
def rational(value) -> Fraction:
    """Preserve the exact value of a finite stored binary64 number."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Exact audit requires finite input numbers")
    return Fraction.from_float(value)


# Compute one DER's allocation probabilities and water level using exact rational arithmetic.
def simplex_response(offers, epsilon):
    """Return exact simplex shares and their unscaled KKT water level."""
    total = ZERO
    water = None
    for count, offer in enumerate(sorted(offers, reverse=True), start=1):
        total += offer
        candidate = (total - epsilon) / count
        if offer > candidate:
            water = candidate
    if water is None:
        raise ArithmeticError("No nonempty simplex support")
    shares = tuple(max(ZERO, (offer - water) / epsilon) for offer in offers)
    if sum(shares, ZERO) != ONE:
        raise ArithmeticError("Exact simplex shares do not sum to one")
    return shares, water


class ExactGame:
    """Original signed MATLAB game with exact, arbitrary-size responses."""

    # Load original MAT coefficients and coalition membership into an independently validated
    # rational model.
    def __init__(self, source: Path):
        stored = loadmat(source, simplify_cells=True)
        parameter = stored["Parameter"]
        f = np.asarray(parameter["FV"])
        if f.ndim != 2:
            raise ValueError("Parameter.FV must be a DER-by-VPP matrix")
        self.d, self.v = f.shape
        if self.d < 1 or self.v < 2 or np.any(f <= 0):
            raise ValueError("Require DERs, at least two VPPs, and positive FV")
        shapes = {
            "FV": (self.d, self.v),
            "CV": (self.d, self.v),
            "Sc": (self.v, self.d),
            "Switching": (self.v, self.d),
            "IV": (self.v,),
        }
        for name, shape in shapes.items():
            values = np.asarray(parameter[name])
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Parameter.{name} must have finite shape {shape}")
        self.f = tuple(tuple(map(rational, row)) for row in parameter["FV"])
        self.c = tuple(tuple(map(rational, row)) for row in parameter["CV"])
        self.sc = tuple(tuple(map(rational, row)) for row in parameter["Sc"])
        self.switch = tuple(tuple(map(rational, row)) for row in parameter["Switching"])
        self.iv = tuple(map(rational, parameter["IV"]))
        self.epsilon = rational(stored["epis"])
        if self.epsilon <= ZERO:
            raise ValueError("epis must be positive")
        incumbent = [-1] * self.d
        entries = np.atleast_1d(stored["cfg"]["vpp"])
        if len(entries) != self.v:
            raise ValueError("cfg.vpp count does not match FV columns")
        for player, entry in enumerate(entries):
            for member in np.atleast_1d(entry["members"]):
                if not np.isfinite(member) or member != int(member):
                    raise ValueError("cfg members must be finite integer indices")
                der = int(member) - 1
                if not 0 <= der < self.d or incumbent[der] != -1:
                    raise ValueError("cfg members must partition DERs exactly once")
                incumbent[der] = player
        if -1 in incumbent:
            raise ValueError("cfg members must cover every DER")
        self.incumbent = tuple(incumbent)

    # Compute exact rational net offers from the supplied VPP strategy profile.
    def offers(self, profile):
        return tuple(
            tuple(
                self.f[d][i] * (ONE - profile[i]) - self.c[d][i] for i in range(self.v)
            )
            for d in range(self.d)
        )

    # Compute exact DER allocation probabilities by applying the rational simplex response.
    def shares(self, profile):
        return tuple(
            simplex_response(row, self.epsilon)[0] for row in self.offers(profile)
        )

    # Evaluate exact payoff directly from the original signed contribution and compensation
    # convention.
    def payoff(self, player, profile, shares=None):
        if shares is None:
            shares = self.shares(profile)
        change = tuple(
            (
                ONE - shares[d][player]
                if self.incumbent[d] == player
                else shares[d][player]
            )
            for d in range(self.d)
        )
        return profile[player] * (
            self.iv[player]
            + sum((self.sc[player][d] * change[d] for d in range(self.d)), ZERO)
        ) + sum((self.switch[player][d] * change[d] for d in range(self.d)), ZERO)

    # Enumerate all unilateral breakpoints and concave vertices to obtain an exact best response.
    def best_response(self, player, profile):
        """Enumerate every support breakpoint and feasible concave vertex."""
        offers = self.offers(profile)
        breakpoints = {ZERO, ONE}
        for der in range(self.d):
            rivals = sorted(
                (offers[der][j] for j in range(self.v) if j != player), reverse=True
            )
            previous = ZERO
            thresholds = []
            for rank, rival in enumerate(rivals, start=1):
                thresholds.append(self.epsilon + rank * rival - previous)
                previous += rival
            thresholds.append(simplex_response(rivals, self.epsilon)[1])
            for threshold in thresholds:
                action = (
                    self.f[der][player] - self.c[der][player] - threshold
                ) / self.f[der][player]
                if ZERO <= action <= ONE:
                    breakpoints.add(action)
        cuts = sorted(breakpoints)
        candidates = set(cuts) | {profile[player]}
        vertex_count = 0
        for left, right in zip(cuts[:-1], cuts[1:]):
            midpoint = (left + right) / 2
            trial = list(profile)
            trial[player] = midpoint
            shares = self.shares(trial)
            quadratic = ZERO
            linear = self.iv[player]
            for der in range(self.d):
                own = shares[der][player]
                support_size = sum(value > ZERO for value in shares[der])
                slope = (
                    -self.f[der][player]
                    * (support_size - 1)
                    / (self.epsilon * support_size)
                    if own > ZERO
                    else ZERO
                )
                intercept = own - slope * midpoint
                if self.incumbent[der] == player:
                    slope, intercept = -slope, ONE - intercept
                quadratic += self.sc[player][der] * slope
                linear += (
                    self.sc[player][der] * intercept + self.switch[player][der] * slope
                )
            if quadratic < ZERO:
                vertex = -linear / (2 * quadratic)
                if left < vertex < right:
                    candidates.add(vertex)
                    vertex_count += 1
        evaluated = []
        for action in sorted(candidates):
            trial = list(profile)
            trial[player] = action
            evaluated.append((self.payoff(player, trial), action))
        value = max(entry[0] for entry in evaluated)
        action = min(entry[1] for entry in evaluated if entry[0] == value)
        return action, value, len(cuts), vertex_count, len(candidates)


# Verify a saved candidate's regret upper bound with exact rational best responses.
def audit(source: Path, candidate_path: Path) -> dict:
    """Return a candidate-only exact audit, with source/candidate fingerprints."""
    candidate_bytes = candidate_path.read_bytes()
    payload = json.loads(candidate_bytes)
    candidate = payload["best_candidate"]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if payload.get("source_sha256", source_hash) != source_hash:
        raise ValueError("Candidate source fingerprint does not match the supplied MAT")
    game = ExactGame(source)
    profile = tuple(map(rational, candidate["x"]))
    if len(profile) != game.v or any(value < ZERO or value > ONE for value in profile):
        raise ValueError("Candidate x must contain one feasible strategy per VPP")
    shares = game.shares(profile)
    profits = [game.payoff(i, profile, shares) for i in range(game.v)]
    responses = [game.best_response(i, profile) for i in range(game.v)]
    regrets = [responses[i][1] - profits[i] for i in range(game.v)]
    if min(regrets) < ZERO:
        raise ArithmeticError("Exact best responses must include staying unchanged")
    maximum_regret = max(regrets)
    return {
        "scope": "Exact candidate regret upper bound only; no global lower-bound audit",
        "coefficient_interpretation": "Exact rational values of original MATLAB binary64 coefficients",
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "candidate_file": str(candidate_path.resolve()),
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "candidate_upper_bound_exactly_verified": True,
        "global_lower_bound_exactly_verified": False,
        "global_nonexistence_exactly_verified": False,
        "x": [float(value) for value in profile],
        "x_rational": list(map(str, profile)),
        "profits": list(map(float, profits)),
        "profits_rational": list(map(str, profits)),
        "best_response_actions": [float(response[0]) for response in responses],
        "best_response_actions_rational": [str(response[0]) for response in responses],
        "best_response_profits": [float(response[1]) for response in responses],
        "best_response_profits_rational": [str(response[1]) for response in responses],
        "regrets": list(map(float, regrets)),
        "regrets_rational": list(map(str, regrets)),
        "maximum_regret": float(maximum_regret),
        "maximum_regret_rational": str(maximum_regret),
        "recorded_float_upper_bound": payload["upper_bound"],
        "maximum_regret_difference_from_record": float(maximum_regret)
        - payload["upper_bound"],
        "max_per_player_regret_difference": max(
            abs(float(regret) - recorded)
            for regret, recorded in zip(regrets, candidate["regrets"])
        ),
        "breakpoint_counts": [response[2] for response in responses],
        "interior_vertex_counts": [response[3] for response in responses],
        "candidate_action_counts": [response[4] for response in responses],
        "shares_rational": [list(map(str, row)) for row in shares],
    }


# Audit the saved candidate, write the requested report, and print its summary.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(args.source, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "maximum_regret",
                    "maximum_regret_difference_from_record",
                    "max_per_player_regret_difference",
                    "breakpoint_counts",
                    "global_lower_bound_exactly_verified",
                )
            },
            indent=2,
        )
    )
    print(f"Audit: {args.output.resolve()}")


if __name__ == "__main__":
    main()
