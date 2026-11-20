"""Check the five-DER certificates with rational arithmetic using Python 3.10+.

Regions are reconstructed by independent half-plane intersection. The checker
enumerates necessary best-response intersections, recomputes global unilateral
regrets and verifies the exact gain polynomial on every witness interval.
 Only the Python standard library is required.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cached_property
from pathlib import Path



Point = tuple[F, F]
Affine = tuple[F, F, F]
Z, O = F(0), F(1)
BOX = ((Z, O, Z), (O, -O, Z), (Z, Z, O), (O, Z, -O))


# Stop verification with a descriptive error when a required condition fails.
def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# Evaluate an affine expression independently for exact geometric and response checks.
def value(line: Affine, point: Point) -> F:
    return line[0] + line[1] * point[0] + line[2] * point[1]


# Reverse all affine coefficients to express the opposite half-plane constraint.
def negate(line: Affine) -> Affine:
    return tuple(-x for x in line)


# Normalize equivalent affine equalities to a common representation for deduplication.
def canonical(line: Affine) -> Affine:
    nonzero = next((x for x in line if x), O)
    return tuple(x / nonzero for x in line)


# Compute the exact intersection of two nonparallel affine equality boundaries.
def intersection(a: Affine, b: Affine) -> Point | None:
    determinant = a[1] * b[2] - a[2] * b[1]
    if determinant == 0:
        return None
    return ((-a[0] * b[2] + b[0] * a[2]) / determinant,
            (-b[0] * a[1] + a[0] * b[1]) / determinant)


# Reconstruct feasible vertices by intersecting boundary pairs and checking every half-plane
# constraint.
def vertices(inequalities: tuple[Affine, ...]) -> tuple[Point, ...]:
    result = set()
    for a, b in itertools.combinations(inequalities, 2):
        point = intersection(a, b)
        if point is not None and all(value(h, point) >= 0 for h in inequalities):
            result.add(point)
    return tuple(sorted(result))


# Recover an exact strategy point at a specified position along a segment.
def interpolate(a: Point, b: Point, t: F) -> Point:
    return tuple((1 - t) * x + t * y for x, y in zip(a, b))


# Recover exact quadratic coefficients from values at the endpoints and midpoint.
def quadratic_from_three(f0: F, fm: F, f1: F) -> tuple[F, F, F]:
    quadratic = 2 * (f1 + f0 - 2 * fm)
    return f0, f1 - f0 - quadratic, quadratic


# Evaluate a quadratic polynomial at the specified exact parameter value.
def quadratic_value(poly: tuple[F, F, F], t: F) -> F:
    return poly[0] + poly[1] * t + poly[2] * t * t


# Independently minimize a quadratic on an interval to check strict deviation gain bounds.
def quadratic_min(poly: tuple[F, F, F], left: F, right: F) -> F:
    positions = [left, right]
    if poly[2] > 0:
        position = -poly[1] / (2 * poly[2])
        if left < position < right:
            positions.append(position)
    return min(quadratic_value(poly, t) for t in positions)


@dataclass(frozen=True)
class Region:
    state: tuple[int, ...]
    halfplanes: tuple[Affine, ...]
    corners: tuple[Point, ...]
    responses: tuple[Affine, ...]


class FrozenGame:
    """Exact game using the original incumbent-leaving probability convention."""

    def __init__(self, inputs: dict):
        owners = inputs["incumbent"]
        require(len(owners) == 5 and all(v in (0, 1) for v in owners), "Invalid incumbent vector")

        def matrix(name, rows, cols):
            data = inputs[name]
            require(len(data) == rows and all(len(row) == cols for row in data), f"Invalid shape for {name}")
            return [[F.from_float(float(entry)) for entry in row] for row in data]

        self.fv = matrix("f", 5, 2)
        self.cv = matrix("c", 5, 2)
        contribution = matrix("contribution", 2, 5)
        self.fees = matrix("compensation", 2, 5)
        self.base = []
        self.sc = []
        for player in range(2):
            own = [d for d, incumbent in enumerate(owners) if incumbent == player]
            intercept = F.from_float(float(inputs["base"][player])) + sum(contribution[player][d] for d in own)
            self.base.append(intercept)
            self.sc.append([-c if owners[d] == player else c for d, c in enumerate(contribution[player])])
            require(all(self.fees[player][d] == 0 for d in range(5) if owners[d] != player),
                    "Departure compensation must go to the incumbent")
        self.epsilon = F.from_float(float(inputs["epsilon"]))
        require(self.epsilon > 0 and all(x > 0 for row in self.fv for x in row), "Invalid response coefficients")
        self.difference = []
        for d, incumbent in enumerate(owners):
            rival = 1 - incumbent
            slope = [Z, Z]
            slope[incumbent] = -self.fv[d][incumbent]
            slope[rival] = self.fv[d][rival]
            self.difference.append((self.fv[d][incumbent] - self.cv[d][incumbent]
                                    - self.fv[d][rival] + self.cv[d][rival], *slope))

    # Independently compute each DER's probability of leaving its incumbent VPP.
    def responses(self, point: Point) -> tuple[F, ...]:
        return tuple(min(O, max(Z, (self.epsilon - value(g, point)) / (2 * self.epsilon)))
                     for g in self.difference)

    # Evaluate an operator's frozen payoff from a supplied vector of leaving probabilities.
    def payoff_from_y(self, player: int, point: Point, y: tuple[F, ...]) -> F:
        return point[player] * (self.base[player] + sum(self.sc[player][d] * y[d] for d in range(5))) + sum(self.fees[player][d] * y[d] for d in range(5))

    # Evaluate the exact payoff using the independently computed leaving probabilities.
    def payoff(self, player: int, point: Point) -> F:
        return self.payoff_from_y(player, point, self.responses(point))

    # Evaluate a fixed region's payoff polynomial, including algebraic extensions used for
    # differentiation.
    def region_payoff(self, player: int, point: Point, region: Region) -> F:
        return self.payoff_from_y(player, point, tuple(value(y, point) for y in region.responses))

    # Reconstruct nonempty response regions through independent half-plane intersection geometry
    # rather than polygon clipping.
    @cached_property
    def regions(self) -> dict[tuple[int, ...], Region]:
        result = {}
        for state in itertools.product(range(3), repeat=5):
            halfplanes, responses = list(BOX), []
            for status, g in zip(state, self.difference):
                if status == 0:
                    halfplanes.append((g[0] - self.epsilon, g[1], g[2]))
                    responses.append((Z, Z, Z))
                elif status == 2:
                    halfplanes.append((-g[0] - self.epsilon, -g[1], -g[2]))
                    responses.append((O, Z, Z))
                else:
                    halfplanes.extend(((self.epsilon - g[0], -g[1], -g[2]),
                                       (self.epsilon + g[0], g[1], g[2])))
                    responses.append(tuple(x / (2 * self.epsilon) for x in
                                           (self.epsilon - g[0], -g[1], -g[2])))
            corners = vertices(tuple(halfplanes))
            if corners:
                result[state] = Region(state, tuple(halfplanes), corners, tuple(responses))
        return result

    # Recover exact affine payoff derivatives through central differences of regional quadratic
    # expressions.
    def derivative(self, player: int, region: Region) -> Affine:
        # Central differences recover the exact derivative of a quadratic.
        # Evaluate an exact coordinate derivative using symmetric evaluations of the regional payoff
        # polynomial.
        def derivative_at(point: Point) -> F:
            plus, minus = list(point), list(point)
            plus[player] += 1
            minus[player] -= 1
            return (self.region_payoff(player, tuple(plus), region)
                    - self.region_payoff(player, tuple(minus), region)) / 2
        constant = derivative_at((Z, Z))
        return constant, derivative_at((O, Z)) - constant, derivative_at((Z, O)) - constant

    # Independently reconstruct necessary equilibrium candidate points, segments, and degenerate
    # regions for coverage comparison.
    def candidates(self):
        points, segments, regions = set(), set(), set()
        for region in self.regions.values():
            player_loci = []
            for player in range(2):
                lines = [BOX[2 * player], BOX[2 * player + 1]]
                for g in self.difference:
                    lines.extend(((g[0] - self.epsilon, g[1], g[2]),
                                  (g[0] + self.epsilon, g[1], g[2])))
                # Include convex stationary points too: harmless overcoverage.
                lines.append(self.derivative(player, region))
                loci = {}
                for line in set(map(canonical, lines)):
                    locus = vertices(region.halfplanes + (line, negate(line)))
                    if locus:
                        loci[line] = locus
                player_loci.append(loci)
            for (a, la), (b, lb) in itertools.product(*(x.items() for x in player_loci)):
                point = intersection(a, b)
                if point is not None:
                    if all(value(h, point) >= 0 for h in region.halfplanes):
                        points.add(point)
                elif a == (Z, Z, Z):
                    locus = lb
                elif b == (Z, Z, Z) or a == b:
                    locus = la
                else:
                    continue
                if point is not None:
                    continue
                if len(locus) == 1:
                    points.add(locus[0])
                elif len(locus) == 2:
                    segments.add(tuple(sorted(locus)))
                else:
                    regions.add(region.state)
        return points, segments, regions

    # Independently compute the global unilateral best-response payoff using breakpoints and exact
    # quadratic interpolation.
    def best_response_value(self, player: int, rival: F) -> F:
        breaks = {Z, O}
        for line in self.difference:
            coefficient = line[player + 1]
            if coefficient == 0:
                continue
            constant = line[0] + line[2 - player] * rival
            for threshold in (-self.epsilon, self.epsilon):
                position = (threshold - constant) / coefficient
                if 0 < position < 1:
                    breaks.add(position)
        breaks = sorted(breaks)

        # Evaluate a proposed own strategy against the rival strategy fixed by the enclosing
        # routine.
        def evaluate(own: F) -> F:
            point = (own, rival) if player == 0 else (rival, own)
            return self.payoff(player, point)

        best = max(map(evaluate, breaks))
        for left, right in zip(breaks, breaks[1:]):
            poly = quadratic_from_three(evaluate(left), evaluate((left + right) / 2), evaluate(right))
            if poly[2] < 0:
                t = -poly[1] / (2 * poly[2])
                if 0 < t < 1:
                    best = max(best, evaluate(left + (right - left) * t))
        return best


# Restore exact strategy coordinates from a certificate's rational fields rather than its displayed
# decimals.
def decode_point(record: dict) -> Point:
    return tuple(map(F, record["rational"]))


# Independently check candidate coverage, exact regrets, and strict segment witnesses without
# generating a new certificate.
def verify(game: FrozenGame, report: dict) -> dict:
    require(F(report["epsilon_exact"]) == game.epsilon, "Certificate epsilon mismatch")
    points, segments, flat_regions = game.candidates()
    reported_points = {decode_point(p["point"]) for p in report["points"]}
    reported_segments = {tuple(sorted(map(decode_point, s["ends"]))) for s in report["segments"]}
    require(points == reported_points, "Necessary isolated-candidate coverage mismatch")
    require(segments == reported_segments, "Necessary segment coverage mismatch")
    require(not flat_regions and not report["unhandled_flat_regions"], "Unverified flat two-dimensional region")
    require(len(game.regions) == report["nonempty_follower_regions"], "Cell count mismatch")
    minimum_point_gain = None
    for record in report["points"]:
        point = decode_point(record["point"])
        require(all(value(h, point) >= 0 for h in BOX), "Point outside strategy box")
        regrets = tuple(game.best_response_value(v, point[1 - v]) - game.payoff(v, point) for v in range(2))
        require(regrets == tuple(map(F, record["regrets_exact"])), "Isolated exact regret mismatch")
        gain = max(regrets)
        require(gain > 0 and record["excluded"], "Isolated candidate not excluded")
        minimum_point_gain = gain if minimum_point_gain is None else min(minimum_point_gain, gain)

    witness_count, smallest_bound, largest_dual = 0, None, Z
    for segment_record in report["segments"]:
        start, end = map(decode_point, segment_record["ends"])
        region = game.regions[tuple(segment_record["state"])]
        require(all(value(h, p) >= 0 for h in region.halfplanes for p in (start, end)), "Segment outside current cell")
        intervals = []
        for witness in segment_record["witnesses"]:
            player = witness["player"] - 1
            require(player in (0, 1), "Invalid witness player")
            left, right = map(F, witness["t"])
            require(0 <= left <= right <= 1, "Invalid witness interval")
            line = tuple(map(F, witness["deviation_line"]))
            require(line[player + 1] != 0, "Invalid deviation line")
            target = game.regions[tuple(witness["deviation_state"])]

            # Reconstruct the strategy implied by a certificate's deviation rule while keeping the
            # rival unchanged.
            def deviation(point: Point) -> Point:
                own = -(line[0] + line[2 - player] * point[1 - player]) / line[player + 1]
                return (own, point[1]) if player == 0 else (point[0], own)

            # Independently recompute a witness's payoff gain at a specified position along its
            # candidate segment.
            def gain(t: F) -> F:
                current = interpolate(start, end, t)
                return game.region_payoff(player, deviation(current), target) - game.region_payoff(player, current, region)

            for t in (left, right):
                proposed = deviation(interpolate(start, end, t))
                require(all(value(h, proposed) >= 0 for h in target.halfplanes), "Witness leaves claimed target cell")
                largest_dual = max(largest_dual, *(max(Z, abs(value(g, proposed)) - game.epsilon) for g in game.difference))
            poly = quadratic_from_three(gain(Z), gain(F(1, 2)), gain(O))
            require(poly == tuple(map(F, witness["gain_polynomial"])), "Witness gain polynomial mismatch")
            bound = quadratic_min(poly, left, right)
            require(bound == F(witness["strict_gain_lower_bound"]) and bound > 0, "Nonpositive or incorrect witness bound")
            smallest_bound = bound if smallest_bound is None else min(smallest_bound, bound)
            intervals.append((left, right))
            witness_count += 1
        covered_until = Z
        for left, right in sorted(intervals):
            require(left <= covered_until, "Open gap in segment witness coverage")
            covered_until = max(covered_until, right)
        require(covered_until == 1 and segment_record["excluded"] and not segment_record["failures"], "Incomplete segment certificate")
        require(len(intervals) == segment_record["witness_count"], "Witness count mismatch")
    require(report["pure_equilibrium_excluded"], "Report does not assert exclusion")
    return {"verified": True, "nonempty_cells": len(game.regions),
            "isolated_candidates": len(points), "candidate_segments": len(segments),
            "verified_witnesses": witness_count,
            "minimum_isolated_max_regret": float(minimum_point_gain),
            "minimum_segment_strict_gain_bound": float(smallest_bound),
            "maximum_dual_needed_by_segment_witnesses": float(largest_dual),
            "scope": "Exact reduced game from stored binary64 coefficients; no parameter robustness claim."}


def full_regrets(game, point):
    return tuple(game.best_response_value(v, point[1-v]) - game.payoff(v, point) for v in (0, 1))


def check_candidate(game, audit):
    point = tuple(map(F, audit["profile_rational"]))
    regrets = full_regrets(game, point)
    require(regrets == tuple(F(p["regret_rational"]) for p in audit["players"]),
            "Reported candidate regrets differ from full-domain best responses")
    return max(regrets)


