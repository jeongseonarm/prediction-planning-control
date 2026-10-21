"""
Frenet local planner as a callable module.

    planner = FrenetLocalPlanner(config, is_collision_fn=collision_checker.check)
    planner.set_waypoints(waypoints)                 # when the global path changes
    traj = planner.plan(ego_state, predictions, obstacle_sizes, obstacle_headings)
    reference = resample_reference(traj, dt, n_points)            # for MPC

Inputs
    waypoints      : (N, >=2) array [x, y, (yaw)] from Hybrid A*
    ego_state      : [x, y, yaw, v, a, steer]  (steer = front wheel angle [rad])
    predictions    : (n_vehicles, K, 2) array or list of (K, 2) arrays [x, y].
                     Rows are spaced config.prediction_dt apart; row 0 is at
                     config.prediction_t0 seconds.
    obstacle_sizes : optional list of (length, width) per vehicle [m]; the
                     config defaults are used where missing.
    obstacle_headings : optional list of the current heading per vehicle [rad],
                     in the same convention as the ego yaw. Used while a vehicle
                     is (almost) standing still; without it such a vehicle is
                     covered by one large circle, which blocks a lot of space.

Collision model
    Ego and vehicles are covered by n_circles circles along their heading.
    The vehicle heading comes from the predicted motion, or from
    obstacle_headings while the vehicle does not move. A candidate is
    rejected when circles come closer than r_ego + r_obstacle + safety_margin.
    Set footprint_collision=False for the old centre-distance check.

Output (always a dict)
    arrays t, x, y, yaw, v, a, kappa, steer
    scalars cost, target_d, target_speed, duration
    status : "ok"     best collision-free candidate
             "reused" no candidate, remaining part of the previous trajectory
             "stop"   no candidate, braking trajectory along the reference
             "min_risk" all candidates collide, the least risky one (beats braking)
    collision_risk : True if a "stop" trajectory still hits a predicted vehicle

When no candidate is feasible the reason decides the fallback:
    collision        -> least risky candidate if it beats braking, else stop
    kinematic / map  -> reuse the previous trajectory for a few cycles, else stop
    range / none     -> stop
Near the end of the reference the target speed is lowered so the vehicle
slows down and stops at the end instead of running out of path.

planner.last_info holds the status, the reason and the rejection counts.
"""

from dataclasses import dataclass

import numpy as np

from .smoothing_reference_path import build_reference_path
from .cartesian_frenet_converter import CartesianFrenetConverter
from .frenet_vectorized import frenet_to_cartesian_batch
from .frenet_trajectory_generator import (
    FrenetTrajectoryGenerator,
    LateralMovement,
    LongitudinalMovement,
)

class _CostOrdered:
    """Candidates of a CandidateSet in ascending cost; paths are built on access."""

    def __init__(self, candidate_set):
        self._set = candidate_set
        self._order = candidate_set.order()

    def __len__(self):
        return len(self._order)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [self._set.path(i) for i in self._order[item]]
        return self._set.path(self._order[item])


_ARRAY_KEYS = ("t", "x", "y", "yaw", "kappa", "v", "a", "steer")


def resample_reference(traj, dt, n_points, keys=("x", "y", "yaw", "v")):
    """
    Sample a planned trajectory at t = 0, dt, 2*dt, ... for a controller whose
    time step is dt (e.g. MPC: n_points = horizon + 1). Past the end of the
    trajectory the last value is held.
    Returns an (n_points, len(keys)) array.
    """
    t = np.arange(n_points) * dt
    return np.column_stack([np.interp(t, traj["t"], traj[k]) for k in keys])


@dataclass
class PlannerConfig:
    # reference
    reference_ds: float = 0.1
    reference_sigma: float = 0.1
    # candidate generation
    dt: float = 0.1
    min_t: float = 1.5
    max_t: float = 2.5
    target_speed: float = 5.0
    # speeds target_speed +- n_speed_samples * target_speed_step (clipped at 0):
    # slower targets let the ego follow a slow vehicle instead of braking to a stop
    target_speed_step: float = 1.0
    n_speed_samples: int = 5
    lateral_offsets: tuple = (-1.5, -0.75, 0.0, 0.75, 1.5)
    # feasibility limits
    max_speed: float = 50.0 / 3.6
    max_accel: float = 10.0
    max_curvature: float = 1.0
    # vehicle and input conventions
    wheelbase: float = 2.7
    prediction_dt: float = 0.04
    # time of the first prediction row [s]: 0.0 if row 0 is the current
    # position, prediction_dt if the rows start one step after it
    prediction_t0: float = 0.04
    # only vehicles whose prediction comes within this distance of the ego
    # are checked [m]
    roi_radius: float = 45.0
    # collision model
    footprint_collision: bool = True
    ego_length: float = 2.5
    ego_width: float = 1.8
    ego_center_offset: float = 0.0   # (x, y) to geometric centre along the heading [m]
    default_obstacle_length: float = 4.5
    default_obstacle_width: float = 1.8
    n_circles: int = 3
    safety_margin: float = 0.3       # extra clearance between footprints [m]
    collision_distance: float = 2.5  # centre distance, only if footprint_collision=False
    # fallbacks
    brake_decel: float = 3.0      # deceleration of the stop trajectory [m/s^2]
    comfort_decel: float = 1.0    # slow-down rate towards the end of the path
    stop_margin: float = 0.5      # stop this far before the reference end [m]
    min_plan_speed: float = 0.3   # below this target speed just stop [m/s]
    max_reuse_cycles: int = 3     # how long the previous trajectory may be reused
    reuse_max_offset: float = 1.5  # ego must be this close to it [m]
    min_reuse_points: int = 5     # shorter remainders are not reused (>= controller horizon)
    risk_time_weight: float = 0.3 # minimum-risk fallback: bonus per second until the first hit [m/s]


class FrenetLocalPlanner:
    def __init__(self, config=None, is_collision_fn=None, is_collision_many_fn=None):
        self.cfg = config or PlannerConfig()
        self._map_blocked = is_collision_fn
        # optional batch version: f(xs, ys, yaws) -> True if ANY pose is blocked
        self._map_blocked_many = is_collision_many_fn
        self.reference = None
        self.last_info = {}
        self._prev = None
        self._age = 0
        self._accel_limit = self.cfg.max_accel
        self.generator = FrenetTrajectoryGenerator(
            lateral_mode=LateralMovement.LOW_SPEED,
            longitudinal_mode=LongitudinalMovement.VELOCITY_KEEPING,
            dt=self.cfg.dt,
            min_t=self.cfg.min_t,
            max_t=self.cfg.max_t,
            target_speed=self.cfg.target_speed,
            target_speed_step=self.cfg.target_speed_step,
            n_speed_samples=self.cfg.n_speed_samples,
            lateral_offsets=self.cfg.lateral_offsets,
        )

    # ------------------------------------------------------------------
    # reference
    # ------------------------------------------------------------------
    def set_waypoints(self, waypoints):
        ref, info = build_reference_path(
            np.asarray(waypoints, dtype=float),
            ds=self.cfg.reference_ds,
            sigma=self.cfg.reference_sigma,
        )
        if len(ref) < 3 or not np.all(np.isfinite(ref)):
            raise ValueError("Invalid reference path.")
        if np.any(np.diff(ref[:, 0]) <= 0):
            raise ValueError("Reference s must be strictly increasing.")

        self.reference = ref
        self.reference_info = info
        self._s = ref[:, 0]
        self._x = ref[:, 1]
        self._y = ref[:, 2]
        self._yaw = np.unwrap(ref[:, 3])
        self._kappa = ref[:, 4]
        self._dkappa = ref[:, 5]

        self._prev = None
        self._age = 0

    def _ref_at(self, s):
        """Linear interpolation of the reference at s (scalar or array)."""
        return (
            np.interp(s, self._s, self._x),
            np.interp(s, self._s, self._y),
            np.interp(s, self._s, self._yaw),
            np.interp(s, self._s, self._kappa),
            np.interp(s, self._s, self._dkappa),
        )

    def _project(self, px, py):
        """Foot of the perpendicular from (px, py) onto the reference polyline."""
        i = int(np.argmin(np.hypot(self._x - px, self._y - py)))
        best = None
        for a, b in ((i - 1, i), (i, i + 1)):
            if a < 0 or b >= len(self._s):
                continue
            p0 = np.array([self._x[a], self._y[a]])
            seg = np.array([self._x[b], self._y[b]]) - p0
            t = np.clip(
                np.dot(np.array([px, py]) - p0, seg) / np.dot(seg, seg), 0.0, 1.0
            )
            dist = np.hypot(*(np.array([px, py]) - (p0 + t * seg)))
            if best is None or dist < best[0]:
                best = (dist, self._s[a] + t * (self._s[b] - self._s[a]))
        return best[1]

    # ------------------------------------------------------------------
    # ego state -> Frenet
    # ------------------------------------------------------------------
    def _ego_to_frenet(self, ego_state):
        x, y, yaw, v, a, steer = (float(u) for u in ego_state)
        kappa = np.tan(steer) / self.cfg.wheelbase

        s_star = self._project(x, y)
        rx, ry, ryaw, rk, rdk = self._ref_at(s_star)

        # keep the heading difference in (-pi, pi]
        yaw = ryaw + np.arctan2(np.sin(yaw - ryaw), np.cos(yaw - ryaw))

        s_state, d_state = CartesianFrenetConverter.cartesian_to_frenet(
            s_star, rx, ry, ryaw, rk, rdk, x, y, v, a, yaw, kappa
        )
        return np.asarray(s_state, dtype=float), np.asarray(d_state, dtype=float)

    # ------------------------------------------------------------------
    # obstacles
    # ------------------------------------------------------------------
    def _circles(self, length, width):
        """Offsets along the heading and radius of the covering circles."""
        n = max(int(self.cfg.n_circles), 1)
        offsets = ((np.arange(n) + 0.5) / n - 0.5) * length
        radius = np.hypot(length / (2 * n), width / 2)
        return offsets, radius

    def _select_obstacles(self, ego_xy, predictions, sizes, headings):
        """Keep vehicles whose predicted path comes within the ROI radius."""
        kept = []
        for i, p in enumerate(predictions):
            p = np.asarray(p, dtype=float)[:, :2]
            if p.ndim != 2 or len(p) == 0 or not np.all(np.isfinite(p)):
                continue
            near = np.min(np.hypot(p[:, 0] - ego_xy[0], p[:, 1] - ego_xy[1]))
            if near > self.cfg.roi_radius:
                continue

            size = None
            if sizes is not None and i < len(sizes) and sizes[i] is not None:
                size = sizes[i]
            if size is None or not np.all(np.isfinite(size)):
                size = (self.cfg.default_obstacle_length, self.cfg.default_obstacle_width)

            heading0 = None
            if headings is not None and i < len(headings) and headings[i] is not None:
                if np.isfinite(headings[i]):
                    heading0 = float(headings[i])
            kept.append((p, float(size[0]), float(size[1]), heading0))
        return kept

    @staticmethod
    def _headings(p, heading0=None, window=5, min_move=0.1):
        """
        Heading per row from the predicted motion. Rows where the vehicle does
        not move far enough use the nearest valid heading, or heading0 before
        the first valid one. Returns (headings, valid); valid is False only if
        there is no motion and no heading0.
        """
        k = len(p)
        ahead = np.minimum(np.arange(k) + window, k - 1)
        delta = p[ahead] - p
        move = np.hypot(delta[:, 0], delta[:, 1])
        heading = np.where(move >= min_move, np.arctan2(delta[:, 1], delta[:, 0]), np.nan)
        moving = ~np.isnan(heading)

        if not moving.any():
            if heading0 is None:
                return np.zeros(k), False
            return np.full(k, float(heading0)), True

        first = int(np.argmax(moving))
        idx = np.maximum.accumulate(np.where(moving, np.arange(k), 0))
        filled = heading[idx]
        filled[:first] = heading0 if heading0 is not None else heading[first]
        return filled, True

    def _obstacles_on_grid(self, obstacles):
        """Resample every kept prediction onto the candidate time grid once."""
        if not obstacles:
            return None

        cfg = self.cfg
        n = int(np.ceil(cfg.max_t / cfg.dt)) + 2
        grid_t = np.arange(n) * cfg.dt

        gx, gy, gyaw, offs, rads = [], [], [], [], []
        for p, length, width, heading0 in obstacles:
            obs_t = cfg.prediction_t0 + np.arange(len(p)) * cfg.prediction_dt
            heading, valid = self._headings(p, heading0)

            gx.append(np.interp(grid_t, obs_t, p[:, 0]))
            gy.append(np.interp(grid_t, obs_t, p[:, 1]))
            gyaw.append(np.interp(grid_t, obs_t, np.unwrap(heading)))

            off, rad = self._circles(length, width)
            if not valid:
                # no usable heading: one bounding circle at the centre
                off = np.zeros_like(off)
                rad = np.hypot(length / 2, width / 2)
            offs.append(off)
            rads.append(rad)

        grid = {
            "x": np.asarray(gx), "y": np.asarray(gy), "yaw": np.asarray(gyaw),
            "off": np.asarray(offs), "rad": np.asarray(rads),
        }
        # obstacle circle centres on the time grid: (M, n_circles, T_grid)
        grid["cx"] = grid["x"][:, None, :] + grid["off"][:, :, None] * np.cos(grid["yaw"])[:, None, :]
        grid["cy"] = grid["y"][:, None, :] + grid["off"][:, :, None] * np.sin(grid["yaw"])[:, None, :]
        return grid

    # ------------------------------------------------------------------
    # Frenet -> Cartesian and checks
    # ------------------------------------------------------------------
    def _to_cartesian(self, path):
        s = np.asarray(path.s, dtype=float)
        rx, ry, ryaw, rk, rdk = self._ref_at(s)
        return frenet_to_cartesian_batch(
            rx, ry, ryaw, rk, rdk, path.s_d, path.s_dd, path.d, path.d_d, path.d_dd
        )  # columns: x, y, yaw, kappa, v, a

    def _kinematic_ok(self, cart):
        return (
            np.all(np.isfinite(cart))
            and np.max(np.abs(cart[:, 4])) <= self.cfg.max_speed
            and np.max(np.abs(cart[:, 5])) <= self._accel_limit
            and np.max(np.abs(cart[:, 3])) <= self.cfg.max_curvature
        )

    def _off_map(self, cart):
        """True if any point of the candidate is blocked on the map."""
        if self._map_blocked_many is not None:
            return bool(self._map_blocked_many(cart[:, 0], cart[:, 1], cart[:, 2]))
        if self._map_blocked is None:
            return False
        return any(
            self._map_blocked(float(x), float(y), float(yaw))
            for x, y, yaw in zip(cart[:, 0], cart[:, 1], cart[:, 2])
        )

    def _collides(self, x, y, yaw, t, grid):
        if grid is None:
            return False

        cfg = self.cfg
        x, y, yaw = np.asarray(x), np.asarray(y), np.asarray(yaw)
        idx = np.clip(
            np.rint(np.asarray(t) / cfg.dt).astype(int), 0, grid["x"].shape[1] - 1
        )
        ox, oy = grid["x"][:, idx], grid["y"][:, idx]

        if not cfg.footprint_collision:
            dist = np.hypot(x[None, :] - ox, y[None, :] - oy)
            return bool(np.min(dist) < cfg.collision_distance)

        # ego circles: (n_ego, T)
        e_off, e_rad = self._circles(cfg.ego_length, cfg.ego_width)
        e_off = e_off + cfg.ego_center_offset
        ex = x[None, :] + e_off[:, None] * np.cos(yaw)[None, :]
        ey = y[None, :] + e_off[:, None] * np.sin(yaw)[None, :]

        # obstacle circles: (M, n_obs, T)
        cx = grid["cx"][:, :, idx]
        cy = grid["cy"][:, :, idx]

        dist = np.hypot(
            cx[:, :, None, :] - ex[None, None, :, :],
            cy[:, :, None, :] - ey[None, None, :, :],
        )
        limit = grid["rad"][:, None, None, None] + e_rad + cfg.safety_margin
        return bool(np.any(dist < limit))

    def _collision_metric(self, x, y, yaw, t, grid):
        """
        Returns (hit, t_hit, min_gap).
        hit     : same decision as _collides (gap < safety_margin somewhere)
        t_hit   : time of the first hit [s], the trajectory duration if none
        min_gap : smallest circle gap over the trajectory without margin [m]
                  (negative = footprints overlap)
        """
        t = np.asarray(t, dtype=float)
        if grid is None:
            return False, float(t[-1]), np.inf

        cfg = self.cfg
        x, y, yaw = np.asarray(x), np.asarray(y), np.asarray(yaw)
        idx = np.clip(np.rint(t / cfg.dt).astype(int), 0, grid["x"].shape[1] - 1)

        e_off, e_rad = self._circles(cfg.ego_length, cfg.ego_width)
        e_off = e_off + cfg.ego_center_offset
        ex = x[None, :] + e_off[:, None] * np.cos(yaw)[None, :]
        ey = y[None, :] + e_off[:, None] * np.sin(yaw)[None, :]
        cx = grid["cx"][:, :, idx]
        cy = grid["cy"][:, :, idx]

        dist = np.hypot(
            cx[:, :, None, :] - ex[None, None, :, :],
            cy[:, :, None, :] - ey[None, None, :, :],
        )
        gap = dist - grid["rad"][:, None, None, None] - e_rad   # (M, n_obs, n_ego, T)
        gap_t = gap.min(axis=(0, 1, 2))                         # (T,)

        hit_idx = np.flatnonzero(gap_t < cfg.safety_margin)
        if hit_idx.size == 0:
            return False, float(t[-1]), float(gap_t.min())
        return True, float(t[hit_idx[0]]), float(gap_t.min())

    def _collision_metric_batch(self, x, y, yaw, t, grid):
        """
        _collision_metric for G trajectories of equal length at once.
        x, y, yaw, t have shape (G, n). Returns (hit, t_hit, min_gap), each (G,).
        """
        G = x.shape[0]
        if grid is None:
            return np.zeros(G, bool), t[:, -1].copy(), np.full(G, np.inf)

        cfg = self.cfg
        idx = np.clip(np.rint(t / cfg.dt).astype(int), 0, grid["x"].shape[1] - 1)

        e_off, e_rad = self._circles(cfg.ego_length, cfg.ego_width)
        e_off = e_off + cfg.ego_center_offset
        ex = x[None] + e_off[:, None, None] * np.cos(yaw)[None]       # (n_ego, G, n)
        ey = y[None] + e_off[:, None, None] * np.sin(yaw)[None]
        cx = grid["cx"][:, :, idx]                                    # (M, n_obs, G, n)
        cy = grid["cy"][:, :, idx]

        dist = np.hypot(
            cx[:, :, None] - ex[None, None],
            cy[:, :, None] - ey[None, None],
        )                                                             # (M, n_obs, n_ego, G, n)
        gap = dist - grid["rad"][:, None, None, None, None] - e_rad
        gap_t = gap.min(axis=(0, 1, 2))                               # (G, n)

        mask = gap_t < cfg.safety_margin
        hit = mask.any(axis=1)
        first = np.argmax(mask, axis=1)
        t_hit = np.where(hit, t[np.arange(G), first], t[:, -1])
        return hit, t_hit, gap_t.min(axis=1)

    def _reachable_grid(self, grid, ego_xy, reach):
        """
        Drop vehicles that no candidate can touch: every candidate point lies
        within `reach` of the ego, so a vehicle whose circles stay farther away
        than reach + ego extent + margin during the whole horizon cannot hit.
        Hits, hit times and gaps of the remaining candidates are unchanged.
        """
        if grid is None:
            return None
        cfg = self.cfg
        e_off, e_rad = self._circles(cfg.ego_length, cfg.ego_width)
        ego_extent = np.max(np.abs(e_off + cfg.ego_center_offset)) + e_rad
        obs_extent = np.max(np.abs(grid["off"]), axis=1) + grid["rad"]            # (M,)
        dmin = np.min(
            np.hypot(grid["x"] - ego_xy[0], grid["y"] - ego_xy[1]), axis=1
        )                                                                          # (M,)
        keep = dmin - obs_extent <= reach + ego_extent + cfg.safety_margin + 1e-6
        if keep.all():
            return grid
        if not keep.any():
            return None
        return {k: v[keep] for k, v in grid.items()}

    def _evaluate_candidates(self, cand, grid, ego_xy):
        """
        Convert and check every candidate in batches (candidates with the same
        number of samples are processed together). Returns per-candidate
        arrays in the order of `cand`; same decisions as checking one by one.
        """
        n_c = len(cand)
        out_of_range = np.zeros(n_c, bool)
        kin_ok = np.zeros(n_c, bool)
        hit = np.zeros(n_c, bool)
        t_hit = np.zeros(n_c)
        gap = np.zeros(n_c)
        carts = [None] * n_c

        groups = {}
        for i, p in enumerate(cand):
            groups.setdefault(len(p.t), []).append(i)

        staged = []
        reach = 0.0
        for idxs in groups.values():
            paths = [cand[i] for i in idxs]
            S = np.stack([p.s for p in paths])
            rx, ry, ryaw, rk, rdk = self._ref_at(S)
            cart = frenet_to_cartesian_batch(
                rx, ry, ryaw, rk, rdk,
                np.stack([p.s_d for p in paths]),
                np.stack([p.s_dd for p in paths]),
                np.stack([p.d for p in paths]),
                np.stack([p.d_d for p in paths]),
                np.stack([p.d_dd for p in paths]),
            )                                                         # (G, n, 6)

            ok = (
                np.isfinite(cart).all(axis=(1, 2))
                & (np.abs(cart[..., 4]).max(axis=1) <= self.cfg.max_speed)
                & (np.abs(cart[..., 5]).max(axis=1) <= self._accel_limit)
                & (np.abs(cart[..., 3]).max(axis=1) <= self.cfg.max_curvature)
            )

            if np.all(np.isfinite(cart)):
                reach = max(
                    reach,
                    float(np.max(np.hypot(cart[..., 0] - ego_xy[0], cart[..., 1] - ego_xy[1]))),
                )
            else:
                reach = np.inf

            ii = np.asarray(idxs)
            out_of_range[ii] = (S.min(axis=1) < self._s[0]) | (S.max(axis=1) > self._s[-1])
            kin_ok[ii] = ok
            staged.append((idxs, ii, cart, np.stack([p.t for p in paths])))
            for row, i in enumerate(idxs):
                carts[i] = cart[row]

        sub_grid = self._reachable_grid(grid, ego_xy, reach)

        for idxs, ii, cart, T in staged:
            h, th, g = self._collision_metric_batch(
                cart[..., 0], cart[..., 1], cart[..., 2], T, sub_grid,
            )
            hit[ii] = h
            t_hit[ii] = th
            gap[ii] = g

        return out_of_range, carts, kin_ok, hit, t_hit, gap

    def _evaluate_in_cost_order(self, ordered, grid, ego_xy, first_chunk=32):
        """
        Yield (path, out_of_range, cart, kin_ok, hit, t_hit, gap) in cost order.
        Candidates are converted and checked in growing chunks, so the work
        stops with the first feasible one. Every candidate gets the same result
        as when all candidates are evaluated at once.
        """
        n = len(ordered)
        start, size = 0, first_chunk
        while start < n:
            paths = ordered[start:start + size]
            out_of_range, carts, kin_ok, hits, t_hits, gaps = self._evaluate_candidates(
                paths, grid, ego_xy
            )
            for j, path in enumerate(paths):
                yield path, out_of_range[j], carts[j], kin_ok[j], hits[j], t_hits[j], gaps[j]
            start += len(paths)
            size *= 2

    def _risk_score(self, t_hit, min_gap):
        # larger is better: more clearance and a later first hit
        return min_gap + self.cfg.risk_time_weight * t_hit

    # ------------------------------------------------------------------
    # fallbacks
    # ------------------------------------------------------------------
    def _stop_trajectory(self, s_state, d_state, grid):
        """Brake along the reference at constant deceleration, lateral offset held."""
        cfg = self.cfg
        s0, v0, _ = s_state
        b = cfg.brake_decel

        t_stop = v0 / b
        n = int(np.ceil(max(cfg.min_t, t_stop) / cfg.dt)) + 1
        t = np.arange(n) * cfg.dt
        tt = np.minimum(t, t_stop)

        s = np.minimum(s0 + v0 * tt - 0.5 * b * tt**2, self._s[-1])
        s_d = np.maximum(v0 - b * t, 0.0)
        s_dd = np.where(t < t_stop, -b, 0.0)

        rx, ry, ryaw, rk, rdk = self._ref_at(s)
        zeros = np.zeros(n)
        cart = frenet_to_cartesian_batch(
            rx, ry, ryaw, rk, rdk, s_d, s_dd,
            np.full(n, d_state[0]), zeros, zeros,
        )

        traj = self._pack(
            t, cart, cost=None, target_d=float(d_state[0]),
            target_speed=0.0, duration=float(t[-1]), status="stop",
        )
        traj["collision_risk"] = self._collides(
            traj["x"], traj["y"], traj["yaw"], traj["t"], grid
        )
        return traj

    def _reuse_previous(self, ego_xy, grid):
        """Remaining part of the previous trajectory, if it is still usable."""
        prev = self._prev
        if prev is None or self._age >= self.cfg.max_reuse_cycles:
            return None

        dist = np.hypot(prev["x"] - ego_xy[0], prev["y"] - ego_xy[1])
        i = int(np.argmin(dist))
        if dist[i] > self.cfg.reuse_max_offset or len(dist) - i < self.cfg.min_reuse_points:
            return None

        traj = dict(prev)
        for key in _ARRAY_KEYS:
            traj[key] = prev[key][i:].copy()
        traj["t"] = traj["t"] - traj["t"][0]
        traj["duration"] = float(traj["t"][-1])
        traj["status"] = "reused"
        traj["collision_risk"] = False

        if self._collides(traj["x"], traj["y"], traj["yaw"], traj["t"], grid):
            return None

        return traj

    def _pack(self, t, cart, cost, target_d, target_speed, duration, status):
        return {
            "t": np.array(t, dtype=float),
            "x": cart[:, 0].copy(),
            "y": cart[:, 1].copy(),
            "yaw": cart[:, 2].copy(),
            "kappa": cart[:, 3].copy(),
            "v": cart[:, 4].copy(),
            "a": cart[:, 5].copy(),
            "steer": np.arctan(self.cfg.wheelbase * cart[:, 3]),
            "cost": cost,
            "target_d": target_d,
            "target_speed": target_speed,
            "duration": duration,
            "status": status,
            "collision_risk": False,
        }

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------
    def min_clearance(self, ego_state, predictions, obstacle_sizes=None, obstacle_headings=None):
        """
        Smallest gap between the ego footprint and the first prediction row
        (about "now") of every vehicle [m], and the index of that vehicle.
        Same circle model as the collision check but without the safety margin;
        a negative value means the footprints overlap.
        """
        cfg = self.cfg
        ex, ey, eyaw = (float(u) for u in ego_state[:3])
        e_off, e_rad = self._circles(cfg.ego_length, cfg.ego_width)
        e_off = e_off + cfg.ego_center_offset
        ecx, ecy = ex + e_off * np.cos(eyaw), ey + e_off * np.sin(eyaw)

        best_gap, best_idx = np.inf, None
        for i, p in enumerate(predictions):
            p = np.asarray(p, dtype=float)[:, :2]
            if p.ndim != 2 or len(p) == 0:
                continue

            size = None
            if obstacle_sizes is not None and i < len(obstacle_sizes):
                size = obstacle_sizes[i]
            if size is None:
                size = (cfg.default_obstacle_length, cfg.default_obstacle_width)
            length, width = float(size[0]), float(size[1])

            heading0 = None
            if obstacle_headings is not None and i < len(obstacle_headings):
                if obstacle_headings[i] is not None and np.isfinite(obstacle_headings[i]):
                    heading0 = float(obstacle_headings[i])

            heading, valid = self._headings(p, heading0)
            off, rad = self._circles(length, width)
            if not valid:
                off = np.zeros_like(off)
                rad = np.hypot(length / 2, width / 2)

            ocx = p[0, 0] + off * np.cos(heading[0])
            ocy = p[0, 1] + off * np.sin(heading[0])
            dist = np.hypot(ocx[:, None] - ecx[None, :], ocy[:, None] - ecy[None, :])
            gap = float(dist.min() - rad - e_rad)

            if gap < best_gap:
                best_gap, best_idx = gap, i

        return best_gap, best_idx

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    def plan(self, ego_state, predictions, obstacle_sizes=None, obstacle_headings=None):
        if self.reference is None:
            raise RuntimeError("Call set_waypoints() first.")

        cfg = self.cfg
        ego_state = np.asarray(ego_state, dtype=float)
        if ego_state.shape != (6,) or not np.all(np.isfinite(ego_state)):
            raise ValueError("ego_state must be a finite [x, y, yaw, v, a, steer].")

        s_state, d_state = self._ego_to_frenet(ego_state)
        s_state[1] = max(s_state[1], 0.0)  # the generator assumes forward motion

        # A candidate starts with the ego's current acceleration. If the ego is
        # already braking harder than max_accel (e.g. after a stop trajectory),
        # that must not make every candidate infeasible.
        self._accel_limit = max(cfg.max_accel, abs(float(ego_state[4])) + 0.5)

        obstacles = self._select_obstacles(
            ego_state[:2], predictions, obstacle_sizes, obstacle_headings
        )
        grid = self._obstacles_on_grid(obstacles)

        # Slow down so that the vehicle can stop before the reference ends.
        remaining = self._s[-1] - s_state[0]
        v_end = np.sqrt(2.0 * cfg.comfort_decel * max(remaining - cfg.stop_margin, 0.0))
        v_target = min(cfg.target_speed, v_end)
        info = {"v_target": float(v_target), "counts": {}}

        if v_target < cfg.min_plan_speed:
            info.update(status="stop", reason="end_of_path")
            self.last_info = info
            return self._finish(self._stop_trajectory(s_state, d_state, grid))

        self.generator.target_speed = v_target
        try:
            if self.generator.supports_batch:
                cand_set = self.generator.generate_batch(s_state=s_state, d_state=d_state)
                candidates = _CostOrdered(cand_set)
            else:
                _, candidates = self.generator.generate(s_state=s_state, d_state=d_state)
                candidates = sorted(candidates, key=lambda p: p.cf)
        except RuntimeError:
            # the generator raises when no candidate can be built, e.g. at standstill
            candidates = []

        counts = {"range": 0, "kinematic": 0, "collision": 0, "map": 0}
        best_risky = None   # (score, path, cart, t_hit, min_gap) among kinematic/map-feasible

        # The first feasible candidate in cost order is the answer.
        stream = self._evaluate_in_cost_order(candidates, grid, ego_state[:2]) if len(candidates) else ()

        for path, out_of_range, cart, kin_ok, hit, t_hit_i, gap_i in stream:
            if out_of_range:
                counts["range"] += 1
                continue

            if not kin_ok:
                counts["kinematic"] += 1
                continue

            if hit:
                t_hit, gap = float(t_hit_i), float(gap_i)
                counts["collision"] += 1
                score = self._risk_score(t_hit, gap)
                if (best_risky is None or score > best_risky[0]) and not self._off_map(cart):
                    best_risky = (score, path, cart, t_hit, gap)
                continue

            if self._off_map(cart):
                counts["map"] += 1
                continue

            traj = self._pack(
                path.t, cart, cost=float(path.cf), target_d=float(path.target_d),
                target_speed=float(path.target_speed),
                duration=float(path.duration), status="ok",
            )
            info.update(status="ok", reason=None, counts=counts)
            self.last_info = info
            return self._finish(traj)

        # ---- no feasible candidate ----
        if counts["collision"] > 0:
            reason = "collision"
        elif counts["kinematic"] > 0:
            reason = "kinematic"
        elif counts["map"] > 0:
            reason = "map"
        elif candidates:
            reason = "range"
        else:
            reason = "no_candidates"

        info.update(reason=reason, counts=counts)

        traj = None
        if reason in ("kinematic", "map"):
            traj = self._reuse_previous(ego_state[:2], grid)

        if traj is None:
            traj = self._stop_trajectory(s_state, d_state, grid)

            # Minimum-risk fallback: every candidate hits something. Use the least
            # bad one if it beats plain braking, otherwise keep braking.
            if reason == "collision" and best_risky is not None:
                _, t_stop_hit, stop_gap = self._collision_metric(
                    traj["x"], traj["y"], traj["yaw"], traj["t"], grid
                )
                score, path, cart, t_hit, gap = best_risky
                if score > self._risk_score(t_stop_hit, stop_gap):
                    traj = self._pack(
                        path.t, cart, cost=float(path.cf), target_d=float(path.target_d),
                        target_speed=float(path.target_speed),
                        duration=float(path.duration), status="min_risk",
                    )
                    traj["collision_risk"] = True
                    info["risk_gap"] = gap
                    info["risk_t_hit"] = t_hit

        info["status"] = traj["status"]
        self.last_info = info
        return self._finish(traj)

    def _finish(self, traj):
        if traj["status"] == "ok":
            self._prev = traj
            self._age = 0
        else:
            self._age += 1
        return traj
