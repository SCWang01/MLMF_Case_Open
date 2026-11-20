"""Numerical interval exclusion for the two-operator frozen market.

For each rectangle, bound every current payoff from above using monotonicity
of each clipped DER share. Bound fixed unilateral quotes from below across
the same rival interval. A positive gain above the guarded tolerance on every
terminal rectangle rules out an approximate equilibrium over the unit square.
"""
import numpy as np
from numba import njit


@njit
def share(intercept, own_slope, rival_slope, own, rival):
    return min(1., max(0., intercept - own_slope * own + rival_slope * rival))


@njit
def box_gain(game, box, player):
    own_low, own_high = box[player]
    rival_low, rival_high = box[1-player]
    coeffs = game[player]
    base = coeffs[0, 5]
    # The share of a given DER decreases in own ratio and increases in rival ratio.
    # Holding the enclosing share intervals fixed gives a convex upper envelope
    # in the own quote, so its maximum occurs at an interval endpoint.
    current_upper = -np.inf
    for own in (own_low, own_high):
        value = own * base
        for intercept, a, b, contribution, fee, _ in coeffs:
            weight = own * contribution - fee
            y = (share(intercept, a, b, own_low, rival_high) if weight >= 0.
                 else share(intercept, a, b, own_high, rival_low))
            value += fee + weight * y
        current_upper = max(current_upper, value)
    gain = -np.inf
    for rival in (rival_low, rival_high, 0.5 * (rival_low + rival_high)):
        for j in range(2 + 2 * len(coeffs)):
            if j < 2:
                quote = float(j)
            else:
                intercept, a, b = coeffs[(j-2)//2, :3]
                quote = min(1., max(0., (intercept + b*rival - (j%2)) / a))
            value = quote * base
            for intercept, a, b, contribution, fee, _ in coeffs:
                weight = quote * contribution - fee
                end = rival_low if weight >= 0. else rival_high
                value += fee + weight * share(intercept, a, b, quote, end)
            gain = max(gain, value-current_upper)
    return gain


@njit
def exclude(game, max_boxes):
    # Depth-first bisection requires one pending sibling per level.
    stack = np.empty((96, 2, 2))
    stack[0] = np.array([[0., 1.], [0., 1.]])
    pending, visits, lower = 1, 0, np.inf
    while pending:
        if visits >= max_boxes:
            return 0., visits
        pending -= 1
        box = stack[pending].copy()
        visits += 1
        gain = box_gain(game, box, 0)
        if gain <= 0.001002:
            gain = max(gain, box_gain(game, box, 1))
        if gain > 0.001002:
            lower = min(lower, gain - 1e-6)
            continue
        coord = int(box[1, 1]-box[1, 0] > box[0, 1]-box[0, 0])
        low, high = box[coord]
        if high-low < 1e-12 or pending >= len(stack)-1:
            return 0., visits
        mid = (low+high)/2
        stack[pending] = box
        stack[pending, coord, 1] = mid
        stack[pending+1] = box
        stack[pending+1, coord, 0] = mid
        pending += 2
    return lower, visits


def coefficients(inputs):
    f, c = np.asarray(inputs['f']), np.asarray(inputs['c'])
    denom = 2*inputs['epsilon']
    game = np.zeros((2, len(f), 6))
    for i in range(2):
        r = 1-i
        game[i,:,0] = 0.5 + (f[:,i]-c[:,i]-f[:,r]+c[:,r])/denom
        game[i,:,1] = f[:,i]/denom
        game[i,:,2] = f[:,r]/denom
        game[i,:,3] = inputs['contribution'][i]
        game[i,:,4] = inputs['compensation'][i]
        game[i,0,5] = inputs['base'][i]
    return game
