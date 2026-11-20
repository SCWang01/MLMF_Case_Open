"""DYNAMICS Simulate the five-DER market over successive switching windows.

Run:  python Supplement/code/dynamics.py Supplement/results/dynamics.json (from the package root)

At every window each VPPO posts the minimax-regret quote of the COPT
rule-enhanced CCG with its refinement settings. A posted quote counts as an
ETA-approximate pure NSE if its verified maximum regret is at most ETA - DELTA.
DERs then choose by the regularized response, contracts age, and the next window
starts from the realized memberships. Fractional choices branch an exact scenario tree.

Returns (JSON written to <output json>):
    schedules.<name>.windows  - per window: P(state mass), expected quotes, probability
                                of meeting the regret threshold (p_pure), expected
                                maximum regret, VPPO cash, DER gain, switches,
                                P(one VPP empty), pruned mass so far
    schedules.<name>.states   - every solved state with its quote and solver record
    provenance                - versions, input hashes and the coefficient rebuild gap

Invariant: the time-varying schedule's first window is the paper's baseline; the
run aborts unless its quote has maximum regret 160.84 (two decimals).
Maintenance: edit only the "Define data" block (schedules, horizon, budget);
coalition values and allocation rules come from market_family.py unchanged.
"""

import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# ---------- Define data ----------------------------------------------------------
# Edit SCHEDULES, WINDOWS or SOLVER to change the experiment; START is the paper's
# five-DER state (order PV1, PV2, TG, ESS, DR = market_family.NAMES).
HERE      = Path(__file__).resolve().parent
FAMILY    = HERE
START     = {"owners": (0, 1, 0, 0, 1), "remaining": (4, 2, 2, 1, 1)}
LENGTHS   = (5, 4)
RULES     = ("equal_dp", "shapley")
EPSILON   = 0.1
GAMMA     = (200.0, 250.0)                 # paper's per-period charge factors
FLAT      = 500.0                          # mean of each VPP's time-varying charge
SCHEDULES = {"time_varying": lambda owner, r: GAMMA[owner] * r,
             "flat":         lambda owner, r: FLAT if r > 0 else 0.0}
WINDOWS   = 10
PRUNE     = 1e-4                           # drop scenario paths below this probability
SHARE_TOL = 1e-9                           # a choice probability below this is zero
ETA       = 1e-3                           # Algorithm 1 tolerance (m.u. per period)
DELTA     = 1e-6
SOLVER    = {"seconds": 300.0, "round_seconds": 30.0, "max_rounds": 200, "threads": 2,
             "absolute_gap": 1e-3, "tau": 0.0, "stop_on_verified": False,
             "policy_mode": "capture", "presolve": 0}     # = baseline.json ccg_settings (COPT refinement)
BASELINE  = 160.84                         # paper's five-DER minimax regret

# ---------- Validate -------------------------------------------------------------
if len(sys.argv) != 2:
    raise SystemExit("dynamics:Usage python dynamics.py <output json>")
OUTPUT = Path(sys.argv[1]).resolve()
if not (FAMILY / "market_family.py").is_file():
    raise FileNotFoundError(f"dynamics:MissingInput {FAMILY / 'market_family.py'}")
if len(START["owners"]) != len(START["remaining"]):
    raise ValueError("dynamics:StartShape owners and remaining terms differ in length")
for d, (owner, r) in enumerate(zip(START["owners"], START["remaining"])):
    if owner not in (0, 1) or not 0 <= r < LENGTHS[owner]:
        raise ValueError(f"dynamics:StartTerm DER {d} owner {owner} remaining {r}")
sys.path.insert(0, str(HERE / "patched"))       # zero-increment patch of the COPT solver
sys.path.insert(1, str(FAMILY))
from importlib.metadata import version                           # noqa: E402
import market_family as mf                      # noqa: E402  (coalition table and rules)
from case20_game import MultiGame               # noqa: E402  (patched copy)
from solve_minimax_4vpp import solve_minimax    # noqa: E402  (patched copy)
D = len(mf.NAMES)
if D != len(START["owners"]):
    raise ValueError(f"dynamics:DERCount market_family has {D} DERs")
ref = json.loads((HERE / "inputs" / "five_baseline.json").read_bytes())["inputs"]
F0, s0, A0 = mf.coefficients((13,18), RULES)
REBUILD_GAP = max(np.max(abs(F0 - ref["f"])), np.max(abs(s0.T - ref["contribution"])), np.max(abs(A0 - ref["base"])))
if REBUILD_GAP > 1e-8:
    raise ValueError("Rebuilt baseline differs from the frozen game")                # coefficients reproduce the saved paper game


# ---------- Derive: one window -----------------------------------------------------
def masks_of(owners):
    return tuple(sum(1 << d for d in range(D) if owners[d] == v) for v in range(2))


def build_game(owners, remaining, schedule):
    """Frozen game of a membership and contract state under one charge schedule."""
    F, s, A = mf.coefficients(masks_of(owners), RULES)
    fee     = [SCHEDULES[schedule](owners[d], remaining[d]) for d in range(D)]
    early   = [remaining[d] > 0 for d in range(D)]
    c = [[fee[d] / LENGTHS[v] if (v != owners[d] and early[d]) else 0.0 for v in range(2)]
         for d in range(D)]
    k = [[fee[d] / remaining[d] if (v == owners[d] and early[d]) else 0.0 for d in range(D)]
         for v in range(2)]
    game = MultiGame(np.asarray(F, dtype=float), np.asarray(c, dtype=float), np.asarray(A, dtype=float),
                     np.asarray(s, dtype=float).T, np.asarray(k, dtype=float),
                     np.asarray(owners, dtype=int), EPSILON)
    return game, fee


def solve_state(game):
    """Minimax-regret quote and its solver record."""
    raw   = solve_minimax(game, progress=None, **SOLVER)
    best  = raw["best_candidate"]
    if best is None:
        raise RuntimeError("dynamics:NoCandidate solver returned no verified profile")
    x     = np.clip(np.asarray(best["x"], dtype=float), 0.0, 1.0)
    regret = float(np.max(game.regrets(x)))
    return {"x": x.tolist(), "max_regret": regret, "pure": regret <= ETA - DELTA,
            "lower": float(raw["lower_bound"]), "upper": float(raw["upper_bound"]),
            "decision": raw["decision"], "stop": raw["stop_reason"],
            "rounds": len(raw["history"]), "seconds": float(raw["elapsed_seconds"])}


def outcomes(game, x):
    """Every realized affiliation with its probability (independent DER choices)."""
    y = game.shares(np.asarray(x))
    supports = []
    for d in range(D):
        support = [(v, float(y[d, v])) for v in range(2) if y[d, v] > SHARE_TOL]
        total   = sum(p for _, p in support)
        supports.append([(v, p / total) for v, p in support])
    for combo in itertools.product(*supports):
        yield tuple(v for v, _ in combo), float(np.prod([p for _, p in combo]))


def realize(owners, remaining, fee, x, new_owners):
    """Cash flows of one realized affiliation and the next contract state."""
    new_masks = masks_of(new_owners)
    early     = [new_owners[d] != owners[d] and remaining[d] > 0 for d in range(D)]
    paid      = [fee[d] if early[d] else 0.0 for d in range(D)]
    cash      = [x[v] * mf.surplus(new_masks[v]) + sum(paid[d] for d in range(D) if owners[d] == v)
                 for v in range(2)]
    shares    = [mf.ALLOCATE[RULES[v]](new_masks[v]) for v in range(2)]
    der_gain  = sum((1 - x[new_owners[d]]) * shares[new_owners[d]][d] - paid[d] for d in range(D))
    renew     = [new_owners[d] != owners[d] or remaining[d] == 0 for d in range(D)]
    next_rem  = tuple((LENGTHS[new_owners[d]] if renew[d] else remaining[d]) - 1 for d in range(D))
    return cash, der_gain, next_rem


# ---------- Derive: the scenario tree ----------------------------------------------
def simulate(schedule):
    states, cache, windows, pruned = {(START["owners"], START["remaining"]): 1.0}, {}, [], 0.0
    for t in range(1, WINDOWS + 1):
        acc  = {"mass": 0.0, "x": np.zeros(2), "pure": 0.0, "max_regret": 0.0, "cash": np.zeros(2),
                "forecast": np.zeros(2), "der_gain": 0.0, "switches": 0.0, "empty_vpp": 0.0}
        nxt  = {}
        for (owners, remaining), mass in states.items():
            key = (owners, remaining)
            game, fee = build_game(owners, remaining, schedule)
            if key not in cache:
                cache[key] = {"owners": list(owners), "remaining": list(remaining), **(first if schedule == "time_varying" and key == (START["owners"], START["remaining"]) else solve_state(game))}
            sol = cache[key]
            x   = sol["x"]
            acc["mass"]       += mass
            acc["x"]          += mass * np.asarray(x)
            acc["pure"]       += mass * sol["pure"]
            acc["max_regret"] += mass * sol["max_regret"]
            acc["forecast"]   += mass * game.payoffs(np.asarray(x))
            acc["empty_vpp"]  += mass * (min(masks_of(owners)) == 0)
            for new_owners, p in outcomes(game, x):
                cash, gain, next_rem = realize(owners, remaining, fee, x, new_owners)
                acc["cash"]     += mass * p * np.asarray(cash)
                acc["der_gain"] += mass * p * gain
                acc["switches"] += mass * p * sum(a != b for a, b in zip(owners, new_owners))
                if mass * p < PRUNE:
                    pruned += mass * p
                    continue
                nxt[(new_owners, next_rem)] = nxt.get((new_owners, next_rem), 0.0) + mass * p
        m = acc["mass"]
        windows.append({"window": t, "mass": m, "n_states": len(states), "pruned_cumulative": pruned,
                        "x": (acc["x"] / m).tolist(), "p_pure": acc["pure"] / m,
                        "max_regret": acc["max_regret"] / m, "vppo_cash": (acc["cash"] / m).tolist(),
                        "vppo_forecast": (acc["forecast"] / m).tolist(), "der_gain": acc["der_gain"] / m,
                        "switches": acc["switches"] / m, "p_empty_vpp": acc["empty_vpp"] / m})
        print(f"{schedule} t={t}: states={len(states)} x={np.round(acc['x'] / m, 3)} "
              f"p_approx={acc['pure'] / m:.3f} regret={acc['max_regret'] / m:.2f} "
              f"cash={np.round(acc['cash'] / m, 1)} der={acc['der_gain'] / m:.1f} pruned={pruned:.2e}",
              flush=True)
        states = nxt
    return {"windows": windows, "states": list(cache.values())}


# ---------- Validate: the first window reproduces the paper's baseline --------------
start = time.perf_counter()
game0, _ = build_game(START["owners"], START["remaining"], "time_varying")
first    = solve_state(game0)
if abs(first["max_regret"] - BASELINE) > 5e-3:
    raise RuntimeError(f"dynamics:BaselineNotReproduced {first['max_regret']:.4f} vs {BASELINE}")
print(f"baseline reproduced: x={np.round(first['x'], 4)} regret={first['max_regret']:.4f}", flush=True)

# ---------- Assemble output ------------------------------------------------------
result = {"schedules": {name: simulate(name) for name in SCHEDULES},
          "settings": {"windows": WINDOWS, "prune": PRUNE, "epsilon": EPSILON, "lengths": LENGTHS,
                       "gamma": GAMMA, "flat": FLAT, "start": START, "solver": SOLVER, "eta": ETA},
          "provenance": {"rebuild_gap": REBUILD_GAP, "coptpy": version("coptpy"),
                         "numpy": np.__version__, "python": platform.python_version(),
                         "processor": platform.processor(), "wall_seconds": time.perf_counter() - start,
                         "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in [FAMILY / "market_family.py", *sorted((HERE / "patched").glob("*.py")),
                                              Path(__file__).resolve()]}}}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=1, allow_nan=False), encoding="utf-8")
