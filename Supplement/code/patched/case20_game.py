"""Big-M-free frozen-window game for the actual four-VPP, twenty-DER case.

The saved Parameter coefficients define the game. Incumbent signs are converted
explicitly rather than replaced by absolute values. Follower responses use the
multi-VPP simplex projection. A unilateral best response enumerates every
support-change threshold and all feasible quadratic maxima; it is a global
one-dimensional optimization, not a gradient or grid approximation.

All player and DER indices in this module are zero-based. Original MAT files
are read only. This numerical module uses binary64; it is not a rational or
interval-arithmetic nonexistence certificate for the four-dimensional game.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "inputs" / "twenty_case.mat"


# Compute regularized simplex response probabilities using sorted offers and water levels.
def simplex_shares(offers, epsilon):
    """Project final-axis offers/epsilon onto a probability simplex."""
    scaled = (offers - np.max(offers, axis=-1, keepdims=True)) / epsilon
    ordered = np.sort(scaled, axis=-1)[..., ::-1]
    cumulative = np.cumsum(ordered, axis=-1) - 1
    count = np.arange(1, ordered.shape[-1] + 1)
    active = ordered - cumulative / count > 0
    rho = active.sum(axis=-1)
    water = np.take_along_axis(cumulative, (rho - 1)[..., None], axis=-1)[..., 0] / rho
    return np.maximum(scaled - water[..., None], 0)


@dataclass(frozen=True)
class Response:
    action: float
    value: float
    candidates: np.ndarray
    values: np.ndarray
    flat_intervals: tuple


@dataclass(frozen=True)
class MultiGame:
    """Validated affine-offer, affine-contribution game with one scalar/player."""

    f: np.ndarray
    c: np.ndarray
    base: np.ndarray
    contribution: np.ndarray
    compensation: np.ndarray
    incumbent: np.ndarray
    epsilon: float = 0.1

    # Validate frozen coefficient shapes, finite values, positive slopes, and incumbent indices.
    def __post_init__(self):
        if self.f.ndim != 2:
            raise ValueError("f must have shape (DER count, VPP count)")
        d, v = self.f.shape
        shapes = {"c": (d, v), "base": (v,), "contribution": (v, d),
                  "compensation": (v, d), "incumbent": (d,)}
        for name, shape in shapes.items():
            value = getattr(self, name)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"Invalid {name}: expected finite array {shape}")
        # PATCH (Supplement dynamics): a VPP with one member or none quotes a zero
        # increment, so zero offer slopes are allowed; negative slopes are not.
        if v < 2 or d < 1 or not np.all(np.isfinite(self.f)) or np.any(self.f < 0):
            raise ValueError("Require at least two VPPs and nonnegative finite offer slopes")
        if self.epsilon <= 0 or not np.isfinite(self.epsilon):
            raise ValueError("epsilon must be positive and finite")
        if np.any(self.incumbent < 0) or np.any(self.incumbent >= v):
            raise ValueError("incumbent indices must identify an existing VPP")

    # Return the number of VPPs represented by the offer matrix columns.
    @property
    def v(self):
        return self.f.shape[1]

    # Return the number of DERs represented by the offer matrix rows.
    @property
    def d(self):
        return self.f.shape[0]

    # Compute net offers to each DER from the supplied VPP strategies.
    def offers(self, x):
        return self.f * (1 - np.asarray(x)[..., None, :]) - self.c

    # Compute DER allocation probabilities from net offers and regularization strength.
    def shares(self, x):
        return simplex_shares(self.offers(x), self.epsilon)

    # Evaluate VPP payoffs using frozen contributions, allocation probabilities, and compensation.
    def payoffs(self, x):
        x = np.asarray(x, dtype=float)
        y = np.swapaxes(self.shares(x), -1, -2)
        return x * (self.base + np.sum(self.contribution * y, axis=-1)) + np.sum(
            self.compensation * (1 - y), axis=-1)

    # Enumerate unilateral strategy breakpoints where follower response slopes can change.
    def own_breakpoints(self, player, x):
        """All possible changes in own affiliation slopes at fixed rivals."""
        rivals = np.delete(self.offers(x), player, axis=1)
        ordered = np.sort(rivals, axis=1)[:, ::-1]
        previous_sum = np.c_[np.zeros(self.d), np.cumsum(ordered, axis=1)[:, :-1]]
        indices = np.arange(1, self.v)
        entrance = self.epsilon + indices * ordered - previous_sum
        rival_shares = simplex_shares(rivals, self.epsilon)
        active = rival_shares > 1e-12
        exit_level = (np.sum(np.where(active, rivals, 0), axis=1) - self.epsilon) / active.sum(axis=1)
        levels = np.c_[entrance, exit_level]
        # PATCH (Supplement dynamics): a zero slope never moves the DER's choice, so
        # its non-finite thresholds are dropped by the [0, 1] filter below.
        with np.errstate(divide="ignore", invalid="ignore"):
            own = (self.f[:, player, None] - self.c[:, player, None] - levels) / self.f[:, player, None]
        return np.unique(np.r_[0.0, 1.0, own[(own >= 0) & (own <= 1)]])

    # Enumerate piecewise candidates to compute a numerical unilateral best response.
    def best_response(self, player, x, previous=None, value_tolerance=1e-7):
        """Global unilateral optimum; ties choose the nearest previous action.

        x contains all current strategies, but x[player] affects only tie
        selection when passed as previous. A flat maximum keeps a previous
        action inside it, instead of silently forcing an interval endpoint.
        """
        x = np.asarray(x, dtype=float)
        if x.shape != (self.v,) or np.any(x < -1e-10) or np.any(x > 1 + 1e-10):
            raise ValueError("x must be a feasible VPP strategy vector")
        if not 0 <= player < self.v:
            raise ValueError("player must be a valid zero-based VPP index")
        cuts = self.own_breakpoints(player, x)
        mids = (cuts[:-1] + cuts[1:]) / 2
        profiles = np.tile(x, (len(mids), 1))
        profiles[:, player] = mids
        y = self.shares(profiles)
        own_share = y[:, :, player]
        support_size = (y > 1e-10).sum(axis=2)
        slope = np.where(own_share > 1e-10,
                         -self.f[:, player] * (support_size - 1) / (self.epsilon * support_size), 0)
        intercept = own_share - slope * mids[:, None]
        quadratic = slope @ self.contribution[player]
        linear = self.base[player] + intercept @ self.contribution[player] - slope @ self.compensation[player]
        stationary = np.divide(-linear, 2 * quadratic, out=np.full_like(linear, np.nan),
                               where=quadratic < -1e-8)
        internal = (stationary > cuts[:-1]) & (stationary < cuts[1:])
        candidates = np.r_[cuts, stationary[internal]]
        flat = (np.abs(quadratic) < 1e-8) & (np.abs(linear) < 1e-7)
        flat_intervals = tuple(zip(cuts[:-1][flat], cuts[1:][flat]))
        if previous is not None:
            candidates = np.r_[candidates, [np.clip(previous, lo, hi) for lo, hi in flat_intervals]]
        candidates = np.unique(candidates)
        candidate_profiles = np.tile(x, (len(candidates), 1))
        candidate_profiles[:, player] = candidates
        values = self.payoffs(candidate_profiles)[:, player]
        top = float(np.max(values))
        tied = candidates[values >= top - value_tolerance]
        action = tied.min() if previous is None else tied[np.argmin(np.abs(tied - previous))]
        return Response(float(action), top, candidates, values, flat_intervals)

    # Compute unilateral regret for each VPP using its complete numerical best response.
    def regrets(self, x):
        return np.array([self.best_response(i, x).value for i in range(self.v)]) - self.payoffs(x)

    # Generate sequential best response trajectories and record strategy, payoff, and regret
    # updates.
    def iterate(self, initial=None, sweeps=1000, order=None, gap_tolerance=1e-6):
        """Return sweep history and every intermediate state with regret checks."""
        x = np.full(self.v, 0.5) if initial is None else np.array(initial, dtype=float)
        order = range(self.v) if order is None else tuple(order)
        history, substeps = [], []
        for k in range(sweeps):
            old = x.copy()
            for player in order:
                response = self.best_response(player, x, previous=x[player])
                x[player] = response.action
                substeps.append([k + 1, player + 1, *x])
            regrets = self.regrets(x)
            history.append([k + 1, *x, *self.payoffs(x), *regrets, np.max(np.abs(x - old))])
            if np.max(regrets) < gap_tolerance:
                break
        return np.asarray(history), np.asarray(substeps)


# Load frozen MAT data and convert signed contributions into allocation probability form.
def load_game(source=SOURCE):
    """Load the actual paper-figure source; top-level Parameter is authoritative."""
    stored = loadmat(source, simplify_cells=True)
    parameters = stored["Parameter"]
    d, v = parameters["FV"].shape
    incumbent = np.full(d, -1, dtype=int)
    for i, entry in enumerate(stored["cfg"]["vpp"]):
        incumbent[np.atleast_1d(entry["members"]).astype(int) - 1] = i
    is_incumbent = np.arange(v)[:, None] == incumbent[None, :]
    signed = np.asarray(parameters["Sc"], dtype=float)
    contribution = np.where(is_incumbent, -signed, signed)
    base = np.asarray(parameters["IV"], dtype=float) + np.sum(np.where(is_incumbent, signed, 0), axis=1)
    game = MultiGame(np.asarray(parameters["FV"], dtype=float),
                     np.asarray(parameters["CV"], dtype=float), base, contribution,
                     np.asarray(parameters["Switching"], dtype=float), incumbent,
                     float(stored["epis"]))
    return game, stored
