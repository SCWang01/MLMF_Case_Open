"""Construct feasible affine deviation rules for arbitrary VPP/DER counts.

The public rule is a tuple ``(intercept, coefficient_0, ..., coefficient_N-1)``
representing ``clip(intercept + coefficients @ x, 0, 1)``. Its deviating player's
coefficient is always zero. The rule matches the supplied best response at the
supplied profile; outside that local support branch it remains a feasible action,
so a regret constraint using its recomputed DER response is globally valid.

The caller must recompute follower responses for the rule's actual action. The
affine share expressions used to construct a rule must not be extended beyond
their support branch as substitutes for those follower responses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from case20_game import MultiGame


ACTION_MATCH_TOLERANCE = 1e-10
SUPPORT_TOLERANCE = 1e-10


# Enumerate affine deviation rules corresponding to follower support switching boundaries.
def _boundary_policies(game: MultiGame, player: int, x: np.ndarray):
    """Yield the affine equations of every own support-change boundary."""
    rivals = np.delete(np.arange(game.v), player)
    offers = game.offers(x)
    for der in range(game.d):
        own_slope = game.f[der, player]
        # PATCH (Supplement dynamics): with a zero own increment the DER's choice
        # does not depend on the own ratio, so it has no own support boundary.
        if own_slope <= 0:
            continue
        own_intercept = own_slope - game.c[der, player]
        ordered = rivals[np.argsort(-offers[der, rivals], kind="stable")]
        previous = []

        # As the own offer falls, the k-th rival enters at
        # q_i = epsilon + k*q_j - sum(previous rivals' offers).
        for rank, rival in enumerate(ordered, start=1):
            coefficients = np.zeros(game.v)
            coefficients[rival] = rank * game.f[der, rival] / own_slope
            coefficients[previous] = -game.f[der, previous] / own_slope
            intercept = (
                own_intercept
                - game.epsilon
                - rank * (game.f[der, rival] - game.c[der, rival])
                + np.sum(game.f[der, previous] - game.c[der, previous])
            ) / own_slope
            yield np.r_[intercept, coefficients]
            previous.append(rival)

        # Own exit occurs at the water level of the rivals-only projection.
        # Shifting offers preserves its active set and reduces cancellation.
        shifted = offers[der, ordered] - offers[der, ordered[0]]
        water = (np.cumsum(shifted) - game.epsilon) / np.arange(1, game.v)
        count = int(np.count_nonzero(shifted > water))
        active = ordered[:count]
        coefficients = np.zeros(game.v)
        coefficients[active] = game.f[der, active] / (count * own_slope)
        intercept = (
            own_intercept
            - (np.sum(game.f[der, active] - game.c[der, active]) - game.epsilon) / count
        ) / own_slope
        yield np.r_[intercept, coefficients]


# Construct an affine rule for a concave payoff stationary point within its response region.
def _stationary_policy(game: MultiGame, player: int, x: np.ndarray, action: float):
    """Derive a quadratic vertex from the support at the supplied own action."""
    profile = x.copy()
    profile[player] = action
    shares = game.shares(profile)
    intercepts = np.zeros(game.d)
    own_slopes = np.zeros(game.d)
    rival_coefficients = np.zeros((game.d, game.v))
    for der in range(game.d):
        if shares[der, player] <= SUPPORT_TOLERANCE:
            continue
        rivals = np.flatnonzero(shares[der] > SUPPORT_TOLERANCE)
        rivals = rivals[rivals != player]
        count = len(rivals)
        denominator = (count + 1) * game.epsilon
        intercepts[der] = (
            count * (game.f[der, player] - game.c[der, player])
            - np.sum(game.f[der, rivals] - game.c[der, rivals])
            + game.epsilon
        ) / denominator
        own_slopes[der] = -count * game.f[der, player] / denominator
        rival_coefficients[der, rivals] = game.f[der, rivals] / denominator

    quadratic = float(game.contribution[player] @ own_slopes)
    if quadratic >= 0:
        return None
    linear = (
        game.base[player]
        + game.contribution[player] @ intercepts
        - game.compensation[player] @ own_slopes
    )
    linear_coefficients = game.contribution[player] @ rival_coefficients
    return np.r_[-linear / (2 * quadratic), -linear_coefficients / (2 * quadratic)]


# Match a best response with a clipped affine rule or constant fallback.
def best_response_policy(
    game: MultiGame, player: int, x, action: float
) -> tuple[float, ...]:
    """Return a clipped affine deviation matching ``action`` at ``x``.

    ``game`` is a validated MultiGame with positive offer slopes; ``x`` and
    ``action`` belong to [0, 1]. Support boundaries are tried before an interior
    quadratic vertex. A constant rule is the safe fallback, including flat
    maxima and numerical ambiguity. No solver or solver license is needed.
    """
    profile = np.asarray(x, dtype=float)
    if profile.shape != (game.v,) or not np.all(np.isfinite(profile)):
        raise ValueError("x must be a finite vector with one strategy per VPP")
    if np.any(profile < 0) or np.any(profile > 1):
        raise ValueError("x must lie in [0, 1]")
    if not isinstance(player, (int, np.integer)) or not 0 <= player < game.v:
        raise ValueError("player must be a valid zero-based VPP index")
    action = float(action)
    if not np.isfinite(action) or not 0 <= action <= 1:
        raise ValueError("action must be a finite number in [0, 1]")

    fallback = np.r_[action, np.zeros(game.v)]
    if action in (0.0, 1.0):
        return tuple(float(value) for value in fallback)

    closest = fallback
    smallest_error = np.inf
    for rule in _boundary_policies(game, player, profile):
        error = abs(float(np.clip(rule[0] + rule[1:] @ profile, 0, 1)) - action)
        if error < smallest_error:
            closest, smallest_error = rule, error
    if smallest_error <= ACTION_MATCH_TOLERANCE:
        return tuple(float(value) for value in closest)

    rule = _stationary_policy(game, player, profile, action)
    if rule is not None and np.all(np.isfinite(rule)):
        error = abs(float(np.clip(rule[0] + rule[1:] @ profile, 0, 1)) - action)
        if error <= ACTION_MATCH_TOLERANCE:
            return tuple(float(value) for value in rule)
    return tuple(float(value) for value in fallback)
