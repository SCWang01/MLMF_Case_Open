"""Rational response checks from the paper's verification/verify_cases.py."""
from __future__ import annotations
from fractions import Fraction as F

Point = tuple[F, F]
Affine = tuple[F, F, F]
Z, O = F(0), F(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def value(line: Affine, point: Point) -> F:
    return line[0] + line[1] * point[0] + line[2] * point[1]


def quadratic_from_three(f0: F, fm: F, f1: F) -> tuple[F, F, F]:
    quadratic = 2 * (f1 + f0 - 2 * fm)
    return f0, f1 - f0 - quadratic, quadratic


class FrozenGame:
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

    def responses(self, point: Point) -> tuple[F, ...]:
        return tuple(min(O, max(Z, (self.epsilon - value(g, point)) / (2 * self.epsilon)))
                     for g in self.difference)

    def payoff_from_y(self, player: int, point: Point, y: tuple[F, ...]) -> F:
        return point[player] * (self.base[player] + sum(self.sc[player][d] * y[d] for d in range(5))) + sum(self.fees[player][d] * y[d] for d in range(5))

    def payoff(self, player: int, point: Point) -> F:
        return self.payoff_from_y(player, point, self.responses(point))

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


def full_regrets(game, point):
    return tuple(game.best_response_value(v, point[1-v]) - game.payoff(v, point) for v in (0, 1))
