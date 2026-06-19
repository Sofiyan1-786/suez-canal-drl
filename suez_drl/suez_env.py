"""
suez_env.py -- Suez Canal Digital Twin (Gymnasium environment).

Discrete-event simulation of one week of Suez Canal vessel scheduling.
Episode: 7 days, 2 convoys/day = 14 steps. Agent picks ships for each convoy.
Reward: -(delay_hours + capital_cost + equity_penalty).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ----------------------------------------------------------------------------
# Realistic cargo distribution (Suez 2024-2025 traffic mix, normalized)
# ----------------------------------------------------------------------------
# (cargo_type, probability, mean_value_usd, std_value_usd, mean_draught_m)
CARGO_MIX = np.array([
    ("bulk",      0.60,  5.0e6,  2.0e6, 14.0),
    ("container", 0.25,  5.0e7,  2.0e7, 13.5),
    ("tanker",    0.10,  3.0e7,  1.5e7, 15.0),
    ("high_val",  0.05,  5.0e8,  2.0e8, 11.0),
], dtype=object)


# ----------------------------------------------------------------------------
# Suez Canal physical parameters (from proposal Section 6.1.1)
# ----------------------------------------------------------------------------
CANAL_MAX_DRAUGHT_M = 20.0          # max allowed ship draught
DEEP_WATER_SPEED_KTS = 14.0         # speed in open water
SQUAT_FACTOR = 0.6                  # v(d) = v0 * (1 - 0.6 * d/d_max)
HOURS_PER_DAY = 24.0
DAYS_PER_EPISODE = 7
CONVOYS_PER_DAY = 2
HOURS_PER_STEP = HOURS_PER_DAY / CONVOYS_PER_DAY  # 12 hours between convoy slots


# ----------------------------------------------------------------------------
# Helper: Squat-limited speed (proposal Eq. for hydrodynamic constraint)
# ----------------------------------------------------------------------------
def squat_speed(draught_m: float) -> float:
    """Max speed (knots) a ship can maintain given its draught in canal."""
    return DEEP_WATER_SPEED_KTS * (1.0 - SQUAT_FACTOR * draught_m / CANAL_MAX_DRAUGHT_M)


# ============================================================================
# Ship — one vessel waiting at Port Said or Suez
# ============================================================================
class Ship:
    __slots__ = (
        "id", "cargo_type", "cargo_value_usd", "wacc",
        "draught_m", "arrival_time_h", "wait_time_h",
    )

    def __init__(self, ship_id: int, rng: np.random.Generator):
        # Sample cargo type by weighted draw
        probs = np.array([row[1] for row in CARGO_MIX], dtype=float)
        probs /= probs.sum()
        idx = rng.choice(len(CARGO_MIX), p=probs)
        row = CARGO_MIX[idx]

        self.id = ship_id
        self.cargo_type = str(row[0])
        # Sample cargo value (lognormal-ish via clipped normal)
        self.cargo_value_usd = max(
            1.0e5,
            float(rng.normal(row[2], row[3])),
        )
        # WACC: capital cost rate, per proposal ~7% +/- 1.5%, clipped
        self.wacc = float(np.clip(rng.normal(0.07, 0.015), 0.03, 0.12))
        # Draught
        self.draught_m = float(np.clip(rng.normal(row[4], 1.5), 8.0, CANAL_MAX_DRAUGHT_M))
        # Hours-from-episode-start at which the ship arrived/arrives
        # in [0, episode_hours]. For ships already waiting at t=0, this is 0.
        self.arrival_time_h = 0.0
        # Time spent waiting so far (hours). Updated each step.
        self.wait_time_h = 0.0

    @property
    def capital_cost_per_hour_usd(self) -> float:
        """$ lost per hour of waiting (value × wacc / 8760 hours/yr)."""
        return self.cargo_value_usd * self.wacc / HOURS_PER_DAY / 365.0

    def features(self) -> np.ndarray:
        """5-dim feature vector used in the observation (all normalized)."""
        return np.array([
            np.log10(self.cargo_value_usd + 1.0) / 9.0,   # 0..1 (5e8→8.7, 1e5→5.0)
            self.wacc,                                      # 0.03..0.12
            self.draught_m / CANAL_MAX_DRAUGHT_M,           # 0..1
            min(self.wait_time_h, 168.0) / 168.0,           # 0..1 (cap at 1 week)
            self.arrival_time_h / (DAYS_PER_EPISODE * HOURS_PER_DAY),
        ], dtype=np.float32)


# ============================================================================
# SuezCanalEnv — main Gymnasium environment
# ============================================================================
class SuezCanalEnv(gym.Env):
    """Suez Canal digital twin for DRL-based vessel scheduling.

    Parameters
    ----------
    n_days : int
        Length of one episode in simulated days (default 7, proposal-aligned).
    convoys_per_day : int
        Number of convoy departures per day (default 2: NB + SB).
    convoy_capacity : int
        Max number of ships per convoy (default 8).
    max_waiting : int
        Maximum number of ships in the queue (env observation/action size).
        Ships beyond this are dropped from observation but counted in IAR.
    arrival_rate_per_day : float
        Poisson mean number of new ships arriving per day (default 5).
    alpha_delay, beta_capital, gamma_equity : float
        Reward coefficients for delay-hours, capital-cost-USD, equity-std.
    seed : int | None
        Random seed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_days: int = DAYS_PER_EPISODE,
        convoys_per_day: int = CONVOYS_PER_DAY,
        convoy_capacity: int = 8,
        max_waiting: int = 20,
        arrival_rate_per_day: float = 5.0,
        alpha_delay: float = 0.01,
        beta_capital: float = 1.0e-3,
        gamma_equity: float = 1.0,
        seed: int | None = None,
        disruption_active: bool = False,
    ):
        super().__init__()
        self.n_days = n_days
        self.convoys_per_day = convoys_per_day
        self.convoy_capacity = convoy_capacity
        self.max_waiting = max_waiting
        self.arrival_rate_per_day = arrival_rate_per_day
        self.alpha_delay = alpha_delay
        self.beta_capital = beta_capital
        self.gamma_equity = gamma_equity
        self.disruption_active = disruption_active

        self.n_steps_per_episode = n_days * convoys_per_day
        self.hours_per_step = HOURS_PER_DAY / convoys_per_day
        self.episode_hours = n_days * HOURS_PER_DAY
        self.arrival_rate_per_step = arrival_rate_per_day / convoys_per_day

        self.rng = np.random.default_rng(seed)

        # Action space: MultiBinary(max_waiting)
        self.action_space = spaces.MultiBinary(max_waiting)

        # Observation space: max_waiting*5 ship features + 3 global features
        n_ship_feats = 5
        n_global_feats = 3
        obs_dim = max_waiting * n_ship_feats + n_global_feats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self._n_ship_feats = n_ship_feats
        self._n_global_feats = n_global_feats

        # State (set in reset)
        self.waiting_queue: list[Ship] = []
        self.current_step: int = 0
        self.current_time_h: float = 0.0
        self.ship_id_counter: int = 0
        self.episode_history: list[dict] = []

    # ---------------------------------------------------------------------
    # Reset
    # ---------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.waiting_queue = []
        self.current_step = 0
        self.current_time_h = 0.0
        self.ship_id_counter = 0
        self.episode_history = []

        # Seed the queue with a few ships already waiting at t=0
        for _ in range(int(self.rng.integers(2, 6))):
            self._add_ship(arrival_time_h=0.0)

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    # ---------------------------------------------------------------------
    # Step
    # ---------------------------------------------------------------------
    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.int8).flatten()
        if action.shape != (self.max_waiting,):
            raise ValueError(
                f"action must have shape ({self.max_waiting},), got {action.shape}"
            )

        # 1. Apply action: pick up to C ships from waiting queue
        n_in_queue = min(len(self.waiting_queue), self.max_waiting)
        chosen_indices = [i for i in range(n_in_queue) if action[i] == 1]
        chosen_indices.sort(key=lambda i: -self.waiting_queue[i].wait_time_h)
        chosen_indices = chosen_indices[: self.convoy_capacity]

        ships_served: list[Ship] = [self.waiting_queue[i] for i in chosen_indices]
        for idx in sorted(chosen_indices, reverse=True):
            self.waiting_queue.pop(idx)

        convoy_departure_h = self.current_time_h + self.hours_per_step

        # 2. Compute reward
        delay_h = sum(
            max(0.0, convoy_departure_h - s.arrival_time_h) for s in ships_served
        )
        capital_cost_usd = sum(
            s.cargo_value_usd * s.wacc * s.wait_time_h / HOURS_PER_DAY / 365.0
            for s in ships_served
        )
        # Equity: std of cargo values of served ships
        if ships_served:
            values = np.array([s.cargo_value_usd for s in ships_served], dtype=np.float64)
            equity_std = float(values.std() / 1.0e8)  # normalize by 100M
        else:
            equity_std = 0.0

        reward = -(
            self.alpha_delay * delay_h
            + self.beta_capital * capital_cost_usd
            + self.gamma_equity * equity_std
        )

        # 3. Log this convoy decision
        self.episode_history.append({
            "step": self.current_step,
            "time_h": convoy_departure_h,
            "n_served": len(ships_served),
            "total_cargo_value_usd": sum(s.cargo_value_usd for s in ships_served),
            "total_capital_cost_usd": capital_cost_usd,
            "total_delay_h": delay_h,
            "n_remaining_in_queue": len(self.waiting_queue),
        })

        # 4. Advance time
        self.current_step += 1
        if self.disruption_active and self.current_step % 2 == 0:
            self.current_time_h += self.hours_per_step * 1.5
        else:
            self.current_time_h += self.hours_per_step

        # 5. New ships arrive (Poisson)
        self._spawn_new_arrivals()

        # 6. Age waiting ships
        for s in self.waiting_queue:
            s.wait_time_h += self.hours_per_step

        # 7. Termination
        terminated = self.current_step >= self.n_steps_per_episode
        truncated = False

        obs = self._build_observation()
        info = self._build_info()
        info["ships_served_this_step"] = len(ships_served)
        info["capital_cost_this_step_usd"] = capital_cost_usd
        info["delay_this_step_h"] = delay_h
        info["equity_std_this_step"] = equity_std
        return obs, float(reward), terminated, truncated, info

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _add_ship(self, arrival_time_h: float):
        s = Ship(self.ship_id_counter, self.rng)
        s.arrival_time_h = arrival_time_h
        self.ship_id_counter += 1
        self.waiting_queue.append(s)

    def _spawn_new_arrivals(self):
        n_new = int(self.rng.poisson(self.arrival_rate_per_step))
        for _ in range(n_new):
            if len(self.waiting_queue) >= self.max_waiting:
                break  # queue full, drop (realistic)
            self._add_ship(arrival_time_h=self.current_time_h)

    def _build_observation(self) -> np.ndarray:
        n = self.max_waiting
        ship_block = np.zeros((n, self._n_ship_feats), dtype=np.float32)
        for i, s in enumerate(self.waiting_queue[:n]):
            ship_block[i] = s.features()

        # Global features
        queue_frac = len(self.waiting_queue) / self.max_waiting
        time_of_day = (self.current_time_h % HOURS_PER_DAY) / HOURS_PER_DAY
        # Current IaR = sum of (cargo_value * wacc * wait_time) for waiting ships
        if self.waiting_queue:
            iar_usd = sum(
                s.cargo_value_usd * s.wacc * s.wait_time_h / HOURS_PER_DAY / 365.0
                for s in self.waiting_queue
            )
        else:
            iar_usd = 0.0
        # log-scale to keep observation in reasonable range
        iar_log = np.log10(iar_usd + 1.0) / 10.0
        global_block = np.array(
            [queue_frac, time_of_day, iar_log], dtype=np.float32
        )

        return np.concatenate([ship_block.flatten(), global_block])

    def _build_info(self) -> dict:
        n = self.max_waiting
        mask = self._compute_action_mask()
        return {
            "action_mask": mask,
            "queue_length": len(self.waiting_queue),
            "time_h": self.current_time_h,
            "step": self.current_step,
        }

    def _compute_action_mask(self) -> np.ndarray:
        n = self.max_waiting
        mask = np.zeros(n, dtype=bool)
        mask[: min(len(self.waiting_queue), n)] = True
        return mask

    # sb3-contrib MaskablePPO interface: flat mask of length 2*max_waiting
    def action_masks(self) -> np.ndarray:
        """Return flat mask of length 2*max_waiting for MaskablePPO.

        Layout: for each ship slot i, [m_0_i, m_1_i, m_0_{i+1}, m_1_{i+1}, ...]
        where m_0 = action 0 (don't pick) validity, m_1 = action 1 (pick) validity.
        A real ship has both True; an empty slot has both False.
        """
        n = self.max_waiting
        per_ship = self._compute_action_mask()  # (n,) bool
        # Interleave: [m_0_0, m_1_0, m_0_1, m_1_1, ...]
        flat = np.empty(2 * n, dtype=bool)
        flat[0::2] = per_ship  # action 0 valid iff slot has a ship
        flat[1::2] = per_ship  # action 1 valid iff slot has a ship
        return flat

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------
    def episode_summary(self) -> dict:
        """Aggregate stats over the just-finished episode."""
        if not self.episode_history:
            return {}
        total_delay = sum(h["total_delay_h"] for h in self.episode_history)
        total_capital = sum(h["total_capital_cost_usd"] for h in self.episode_history)
        total_served = sum(h["n_served"] for h in self.episode_history)
        total_cargo = sum(h["total_cargo_value_usd"] for h in self.episode_history)
        return {
            "total_delay_h": float(total_delay),
            "total_capital_cost_usd": float(total_capital),
            "total_ships_served": int(total_served),
            "total_cargo_value_usd": float(total_cargo),
            "n_steps": len(self.episode_history),
        }


# ============================================================================
# Self-test (run directly: `python suez_env.py`)
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SuezCanalEnv self-test: 5 random episodes")
    print("=" * 60)
    env = SuezCanalEnv(seed=42)
    for ep in range(5):
        obs, info = env.reset(seed=ep)
        print(f"\n--- Episode {ep+1} ---")
        print(f"  init queue={info['queue_length']}, mask sum={info['action_mask'].sum()}")
        ep_reward = 0.0
        done = False
        steps = 0
        while not done:
            mask = info["action_mask"]
            # Random valid action: pick 0..convoy_capacity ships at random from masked slots
            valid_idx = np.where(mask)[0]
            action = np.zeros(env.max_waiting, dtype=np.int8)
            k = int(env.rng.integers(0, min(env.convoy_capacity, len(valid_idx)) + 1))
            if k > 0:
                chosen = env.rng.choice(valid_idx, size=k, replace=False)
                action[chosen] = 1
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            steps += 1
            done = terminated or truncated
        summary = env.episode_summary()
        print(f"  steps={steps}, ep_reward={ep_reward:+.2f}")
        print(f"  total_delay_h        = {summary['total_delay_h']:8.2f}")
        print(f"  total_capital_cost   = ${summary['total_capital_cost_usd']:,.2f}")
        print(f"  total_ships_served   = {summary['total_ships_served']}")
        print(f"  total_cargo_value    = ${summary['total_cargo_value_usd']:,.0f}")
    print("\nOK — env runs cleanly with random actions.")
