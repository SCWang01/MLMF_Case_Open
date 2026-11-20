"""Recover exact five-DER pure equilibria near numerical quotes and verify global best responses."""
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "small"))
from exact_audit import FrozenGame, full_regrets, value

ZERO = Fraction(0)
ONE = Fraction(1)
BOX = ((ZERO, ONE, ZERO), (-ONE, ONE, ZERO), (ZERO, ZERO, ONE), (-ONE, ZERO, ONE))


def intersection(first, second):
    a, b, c = first
    d, e, f = second
    det = b * f - c * e
    if det:
        return ((c * d - a * f) / det, (a * e - b * d) / det)


def distance(line, point):
    scale = max(abs(float(line[1])), abs(float(line[2])))
    return abs(float(value(line, point))) / scale if scale else float("inf")


def nearby_lines(game, point, radius):
    """Boundary and smooth stationary lines from response states near the quoted profile."""
    lines = {line for line in BOX if distance(line, point) <= radius}
    states = []
    eps = game.epsilon
    for g in game.difference:
        at = value(g, point)
        statuses = {0 if at >= eps else 2 if at <= -eps else 1}
        for threshold, outer in ((eps, 0), (-eps, 2)):
            boundary = (g[0] - threshold, *g[1:])
            if distance(boundary, point) <= radius:
                lines.add(boundary)
                statuses.update((outer, 1))
        states.append(statuses)
    for state in product(*states):
        for p in range(2):
            own = p + 1
            deriv = [game.base[p], ZERO, ZERO]
            for d, status in enumerate(state):
                s = game.sc[p][d]
                if status == 2:
                    deriv[0] += s
                elif status == 1:
                    g = game.difference[d]
                    y = ((eps - g[0])/(2*eps), -g[1]/(2*eps), -g[2]/(2*eps))
                    deriv[0] += s*y[0] + game.fees[p][d]*y[own]
                    deriv[1] += s*y[1]
                    deriv[2] += s*y[2]
                    deriv[own] += s*y[own]
            if deriv[own] <= 0 and distance(deriv, point) <= radius:
                lines.add(tuple(deriv))
    return lines


def propose(game, point, radius):
    lines = nearby_lines(game, point, radius)
    candidates = set()
    for first, second in combinations(lines, 2):
        hit = intersection(first, second)
        if hit is not None:
            candidates.add(hit)
    # A common optimal breakpoint may carry a whole interval of equilibria.
    # Preserve one reported coordinate and solve exactly for the other.
    for a, b, c in lines:
        if b:
            candidates.add((-(a + c*point[1])/b, point[1]))
        if c:
            candidates.add((point[0], -(a + b*point[0])/c))
    feasible = (x for x in candidates if all(0 <= v <= 1 for v in x))
    return sorted(feasible, key=lambda x: (max(abs(float(a-b)) for a,b in zip(point,x)), x))


def witness(game, point):
    # Reject a nonoptimal candidate as soon as one player has a profitable deviation.
    return all(game.best_response_value(p, point[1-p]) == game.payoff(p, point) for p in range(2))


def exact_check(inputs, quote):
    """Return verified rational coordinates, or None when no nearby exact witness is found."""
    if quote is None:
        return None
    point = tuple(map(Fraction, quote))
    if len(point) != 2 or not all(0 <= x <= 1 for x in point):
        raise ValueError("Expected a feasible two-player quote.")
    game = FrozenGame(inputs)
    tested = {point}
    if witness(game, point):
        return list(map(str, point))
    for radius in (1e-8, 2e-5, 0.001):
        for trial in propose(game, point, radius):
            if trial not in tested and witness(game, trial):
                return list(map(str, trial))
            tested.add(trial)
    return None

