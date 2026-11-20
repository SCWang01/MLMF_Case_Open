"""Generate exact point and segment proofs for the released five-DER games."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
from typing import Iterable



Point = tuple[Q, Q]
Affine = tuple[Q, Q, Q]  # Constant, x1, x2.
Polynomial = tuple[Q, Q, Q]  # Constant, t, t squared.
ZERO = Q(0)
ONE = Q(1)
SQUARE = [(ZERO, ZERO), (ONE, ZERO), (ONE, ONE), (ZERO, ONE)]


@dataclass(frozen=True)
class Cell:
    state: tuple[int, ...]
    polygon: tuple[Point, ...]
    inequalities: tuple[Affine, ...]  # All expressions are nonnegative.
    y: tuple[Affine, ...]


@dataclass(frozen=True)
class Deviation:
    player: int
    line: Affine
    cell: Cell


# Convert a stored binary64 value into its exact rational representation.

def affine_value(line: Affine, point: Point) -> Q:
    return line[0] + line[1] * point[0] + line[2] * point[1]


# Construct an exact point along the segment between two strategy profiles.
def interpolate(a: Point, b: Point, weight: Q) -> Point:
    return tuple(a[i] + weight * (b[i] - a[i]) for i in range(2))


# Remove duplicate points while preserving their order for subsequent polygon operations.
def unique_points(points: Iterable[Point]) -> list[Point]:
    result = []
    for point in points:
        if point not in result:
            result.append(point)
    return result


# Clip a polygon to the feasible side of an affine inequality.
def clip_polygon(polygon: list[Point], inequality: Affine) -> list[Point]:
    if not polygon:
        return []
    clipped = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        fa, fb = affine_value(inequality, a), affine_value(inequality, b)
        if fa >= 0:
            clipped.append(a)
        if (fa < 0 < fb) or (fb < 0 < fa):
            clipped.append(interpolate(a, b, fa / (fa - fb)))
    return unique_points(clipped)


# Find the exact intersection of an affine equality with a response polygon.
def line_polygon(line: Affine, polygon: tuple[Point, ...]) -> tuple[Point, ...]:
    if line == (ZERO, ZERO, ZERO):
        return polygon
    found = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        fa, fb = affine_value(line, a), affine_value(line, b)
        if fa == 0:
            found.append(a)
        if (fa < 0 < fb) or (fb < 0 < fa):
            found.append(interpolate(a, b, fa / (fa - fb)))
    found = sorted(set(found))
    return tuple(found if len(found) <= 2 else (found[0], found[-1]))


class Game:
    """Validated two-leader game with exact follower and payoff evaluations."""

    # Build exact response regions and deviations from the released coefficients.
    def __init__(self, rec: dict):
        D = len(rec["incumbent"])
        q = Q.from_float
        inc = rec["incumbent"]
        f   = [[q(float(z)) for z in row] for row in rec["f"]]
        c   = [[q(float(z)) for z in row] for row in rec["c"]]
        s   = [[q(float(z)) for z in row] for row in rec["contribution"]]
        k   = [[q(float(z)) for z in row] for row in rec["compensation"]]
        A   = [q(float(z)) for z in rec["base"]]
        if any(z <= 0 for row in f for z in row):
            raise ValueError("exact_game:NonPositiveSlope the exact auditor requires F_dv > 0")
        self.fv     = tuple(tuple(row) for row in f)
        self.cv     = tuple(tuple(row) for row in c)
        self.iv     = tuple(A[p] + sum(s[p][d] for d in range(D) if inc[d] == p) for p in range(2))
        self.sc     = tuple(tuple(-s[p][d] if inc[d] == p else s[p][d] for d in range(D)) for p in range(2))
        self.switch = tuple(tuple(k[p][d] if inc[d] == p else Q(0) for d in range(D)) for p in range(2))
        self.eps    = q(float(rec["epsilon"]))
        signs       = [1 if inc[d] == 0 else -1 for d in range(D)]
        self.g      = tuple((sg * (fd[0] - cd[0] - fd[1] + cd[1]), -sg * fd[0], sg * fd[1])
                            for sg, fd, cd in zip(signs, self.fv, self.cv))
        self.cells      = self._cells()
        self.deviations = self._deviations()

    def response(self, point: Point) -> tuple[Q, ...]:
        return tuple(
            max(ZERO, min(ONE, (self.eps - affine_value(g, point)) / (2 * self.eps)))
            for g in self.g
        )

    # Evaluate one operator's frozen payoff, including compensation, using the current leaving
    # probabilities.
    def payoff(self, player: int, point: Point) -> Q:
        y = self.response(point)
        return point[player] * (self.iv[player] + sum(s * z for s, z in zip(self.sc[player], y))) + sum(w * z for w, z in zip(self.switch[player], y))

    # Find a global unilateral best response by comparing endpoints, response breakpoints, and
    # concave stationary points.
    def best_response(self, player: int, rival: Q) -> tuple[Q, Q]:
        a = [g[player + 1] for g in self.g]
        b = [g[0] + g[2 - player] * rival for g in self.g]
        cuts = sorted({ZERO, ONE} | {
            max(ZERO, min(ONE, (threshold - bb) / aa))
            for aa, bb in zip(a, b) for threshold in (-self.eps, self.eps)
        })
        candidates = list(cuts)
        for left, right in zip(cuts[:-1], cuts[1:]):
            midpoint = (left + right) / 2
            slopes, intercepts = [], []
            for aa, bb in zip(a, b):
                value = aa * midpoint + bb
                if -self.eps < value < self.eps:
                    slopes.append(-aa / (2 * self.eps))
                    intercepts.append((self.eps - bb) / (2 * self.eps))
                else:
                    slopes.append(ZERO)
                    intercepts.append(ONE if value < 0 else ZERO)
            quad = sum(s * m for s, m in zip(self.sc[player], slopes))
            linear = self.iv[player] + sum(s * c for s, c in zip(self.sc[player], intercepts)) + sum(w * m for w, m in zip(self.switch[player], slopes))
            if quad < 0:
                vertex = -linear / (2 * quad)
                if left < vertex < right:
                    candidates.append(vertex)
        results = []
        for own in candidates:
            point = (own, rival) if player == 0 else (rival, own)
            results.append((self.payoff(player, point), own))
        value, own = max(results)
        return own, value

    # Compute both operators' exact gains from their respective global unilateral best responses.
    def regrets(self, point: Point) -> tuple[Q, Q]:
        return tuple(self.best_response(i, point[1 - i])[1] - self.payoff(i, point) for i in range(2))

    # Enumerate nonempty response regions and retain their affine leaving probability expressions.
    def _cells(self) -> tuple[Cell, ...]:
        cells = []
        square_inequalities = ((ZERO, ONE, ZERO), (ONE, -ONE, ZERO),
                               (ZERO, ZERO, ONE), (ONE, ZERO, -ONE))
        for state in itertools.product(range(3), repeat=5):
            inequalities = list(square_inequalities)
            y = []
            for status, g in zip(state, self.g):
                if status == 0:
                    inequalities.append((g[0] - self.eps, g[1], g[2]))
                    y.append((ZERO, ZERO, ZERO))
                elif status == 2:
                    inequalities.append((-self.eps - g[0], -g[1], -g[2]))
                    y.append((ONE, ZERO, ZERO))
                else:
                    inequalities.extend(((self.eps - g[0], -g[1], -g[2]),
                                         (self.eps + g[0], g[1], g[2])))
                    y.append(((self.eps - g[0]) / (2 * self.eps),
                              -g[1] / (2 * self.eps), -g[2] / (2 * self.eps)))
            polygon = list(SQUARE)
            for inequality in inequalities:
                polygon = clip_polygon(polygon, inequality)
                if not polygon:
                    break
            if polygon:
                cells.append(Cell(state, tuple(polygon), tuple(inequalities), tuple(y)))
        return tuple(cells)

    # Identify boundary, breakpoint, and stationary conditions that can contain unilateral optima
    # within a region.
    def necessary_lines(self, player: int, cell: Cell) -> tuple[Affine, ...]:
        # A coordinate optimum is a boundary, a response kink, or a stationary
        # point in a smooth nonconvex-downward (including flat) payoff segment.
        own = player + 1
        base = [ZERO, ZERO, ZERO]
        base[own] = ONE
        lines = [tuple(base), (-ONE, base[1], base[2])]
        for g in self.g:
            lines.extend(((g[0] - self.eps, g[1], g[2]),
                          (g[0] + self.eps, g[1], g[2])))
        derivative = [self.iv[player], ZERO, ZERO]
        for s, w, y in zip(self.sc[player], self.switch[player], cell.y):
            derivative[0] += s * y[0] + w * y[own]
            derivative[1] += s * y[1]
            derivative[2] += s * y[2]
            derivative[own] += s * y[own]
        if derivative[own] <= 0:
            lines.append(tuple(derivative))
        return tuple(line for line in dict.fromkeys(lines) if line_polygon(line, cell.polygon))

    # Construct candidate deviation rules from regional conditions that determine an operator's own
    # strategy.
    def _deviations(self) -> tuple[Deviation, ...]:
        pieces = []
        for player in range(2):
            for cell in self.cells:
                for line in self.necessary_lines(player, cell):
                    if line[player + 1] != 0:
                        pieces.append(Deviation(player, line, cell))
        return tuple(pieces)


# Intersect two necessary optimality conditions within a response region, including coincident or
# degenerate cases.
def intersect_lines(first: Affine, second: Affine, cell: Cell) -> tuple[Point, ...]:
    if first == (ZERO, ZERO, ZERO):
        return line_polygon(second, cell.polygon)
    if second == (ZERO, ZERO, ZERO):
        return line_polygon(first, cell.polygon)
    determinant = first[1] * second[2] - first[2] * second[1]
    if determinant == 0:
        if first[0] * second[1] == second[0] * first[1] and first[0] * second[2] == second[0] * first[2]:
            return line_polygon(first, cell.polygon)
        return ()
    point = ((first[2] * second[0] - first[0] * second[2]) / determinant,
             (first[0] * second[1] - first[1] * second[0]) / determinant)
    return (point,) if all(affine_value(h, point) >= 0 for h in cell.inequalities) else ()


# Expand an operator's regional payoff along a strategy segment into exact quadratic coefficients.
def payoff_polynomial(game: Game, player: int, cell: Cell, start: Point, end: Point) -> Polynomial:
    direction = tuple(end[i] - start[i] for i in range(2))
    b0, b1, w0, w1 = game.iv[player], ZERO, ZERO, ZERO
    for s, w, y in zip(game.sc[player], game.switch[player], cell.y):
        intercept = affine_value(y, start)
        slope = y[1] * direction[0] + y[2] * direction[1]
        b0 += s * intercept
        b1 += s * slope
        w0 += w * intercept
        w1 += w * slope
    return (start[player] * b0 + w0,
            start[player] * b1 + direction[player] * b0 + w1,
            direction[player] * b1)


# Find a quadratic polynomial's exact minimum over an interval using endpoints and any interior
# minimum.
def polynomial_minimum(poly: Polynomial, left: Q, right: Q) -> Q:
    # Evaluate the enclosing quadratic polynomial at a candidate interval position.
    def value(t: Q) -> Q:
        return poly[0] + t * (poly[1] + t * poly[2])
    result = min(value(left), value(right))
    if poly[2] > 0:
        vertex = -poly[1] / (2 * poly[2])
        if left < vertex < right:
            result = min(result, value(vertex))
    return result


# Determine where a deviation remains feasible and compute its payoff gain along the current
# segment.
def deviation_along_segment(game: Game, deviation: Deviation, cell: Cell,
                            start: Point, end: Point):
    player, line = deviation.player, deviation.line
    # Replace one operator's strategy using the deviation rule while preserving the rival's
    # strategy.
    def deviated(point: Point) -> Point:
        own = -(line[0] + line[2 - player] * point[1 - player]) / line[player + 1]
        return (own, point[1]) if player == 0 else (point[0], own)
    a, b = deviated(start), deviated(end)
    left, right = ZERO, ONE
    for inequality in deviation.cell.inequalities:
        intercept = affine_value(inequality, a)
        slope = affine_value(inequality, b) - intercept
        if slope > 0:
            left = max(left, -intercept / slope)
        elif slope < 0:
            right = min(right, -intercept / slope)
        elif intercept < 0:
            return None
    if left > right:
        return None
    proposed = payoff_polynomial(game, player, deviation.cell, a, b)
    current = payoff_polynomial(game, player, cell, start, end)
    gain = tuple(proposed[i] - current[i] for i in range(3))
    return left, right, gain, deviation


# Serialize a rational number as an exact numerator and denominator string.
def encode_q(value: Q) -> str:
    return f"{value.numerator}/{value.denominator}"


# Record both readable decimal coordinates and exact rational coordinates for a strategy profile.
def encode_point(point: Point) -> dict:
    return {"float": list(map(float, point)), "rational": list(map(encode_q, point))}


# Build profitable deviation witnesses covering a candidate segment, recording any unresolved
# portions.
def certify_segment(game: Game, cell: Cell, segment: tuple[Point, Point]) -> dict:
    pieces = [piece for deviation in game.deviations
              if (piece := deviation_along_segment(game, deviation, cell, *segment)) is not None]
    cuts = sorted({ZERO, ONE} | {bound for piece in pieces for bound in piece[:2]})
    witnesses = []
    failures = []
    smallest_gap = None

    # Cover an interval with strict deviation gains, subdividing when necessary and recording
    # equilibrium or depth-limit failures.
    def cover(left: Q, right: Q, valid: list, depth: int = 0):
        nonlocal smallest_gap
        gains = [(polynomial_minimum(piece[2], left, right), piece) for piece in valid]
        bound, selected = max(gains, key=lambda entry: entry[0])
        if bound > 0:
            smallest_gap = bound if smallest_gap is None else min(smallest_gap, bound)
            witnesses.append({"t": [encode_q(left), encode_q(right)],
                              "player": selected[3].player + 1,
                              "deviation_line": list(map(encode_q, selected[3].line)),
                              "deviation_state": selected[3].cell.state,
                              "gain_polynomial": list(map(encode_q, selected[2])),
                              "strict_gain_lower_bound": encode_q(bound)})
            return
        midpoint = (left + right) / 2
        point = interpolate(*segment, midpoint)
        regrets = game.regrets(point)
        if max(regrets) == 0 or depth >= 60:
            failures.append({"point": encode_point(point), "regrets": list(map(float, regrets)),
                             "depth": depth, "exact_equilibrium": max(regrets) == 0})
            return
        cover(left, midpoint, valid, depth + 1)
        cover(midpoint, right, valid, depth + 1)

    for left, right in zip(cuts[:-1], cuts[1:]):
        valid = [piece for piece in pieces if piece[0] <= left and piece[1] >= right]
        if not valid:
            raise RuntimeError("Missing global deviation candidates on a segment interval")
        cover(left, right, valid)
    return {"state": cell.state, "ends": [encode_point(p) for p in segment],
            "witness_count": len(witnesses), "witnesses": witnesses,
            "minimum_certified_gain": float(smallest_gap) if smallest_gap else None,
            "failures": failures, "excluded": not failures}


# Check the baseline ESS-retention response curve for global optimality, uniqueness, and profitable
# deviations along it.
def audit(game: Game) -> dict:
    points = {}
    segments = {}
    regions = []
    for cell in game.cells:
        for first, second in itertools.product(game.necessary_lines(0, cell), game.necessary_lines(1, cell)):
            locus = intersect_lines(first, second, cell)
            if len(locus) == 1:
                points[locus[0]] = cell.state
            elif len(locus) == 2:
                segments[tuple(sorted(locus))] = cell
            elif len(locus) > 2:
                regions.append(cell.state)
    print(f"Regions: {len(game.cells)}; isolated candidates: {len(points)}; candidate segments: {len(segments)}", flush=True)
    point_results = []
    for point, state in points.items():
        regrets = game.regrets(point)
        point_results.append({"point": encode_point(point), "state": state,
                              "regrets": list(map(float, regrets)),
                              "regrets_exact": list(map(encode_q, regrets)),
                              "excluded": max(regrets) > 0})
    print(f"Isolated candidates excluded: {sum(p['excluded'] for p in point_results)}", flush=True)
    segment_results = []
    for index, (segment, cell) in enumerate(segments.items(), 1):
        result = certify_segment(game, cell, segment)
        segment_results.append(result)
        print(f"Segment {index}/{len(segments)}: excluded={result['excluded']}, witnesses={result['witness_count']}", flush=True)
    excluded = not regions and all(p["excluded"] for p in point_results) and all(s["excluded"] for s in segment_results)
    return {"arithmetic": "exact fractions of original binary64 MAT coefficients",
            "epsilon_exact": encode_q(game.eps), "nonempty_follower_regions": len(game.cells),
            "isolated_candidate_count": len(points), "candidate_segment_count": len(segments),
            "unhandled_flat_regions": regions, "pure_equilibrium_excluded": excluded,
            "minimum_isolated_candidate_max_regret": min(max(p["regrets"]) for p in point_results),
            "points": point_results, "segments": segment_results}


