"""
Frenet trajectory candidate generator.

Quartic s(t) / quintic d(s) candidates and the jerk-time-offset cost follow
Werling et al., "Optimal Trajectory Generation for Dynamic Street Scenarios in
a Frenet Frame" (2010), in the form used by PythonRobotics (Atsushi Sakai, MIT).

generate()        any lateral / longitudinal mode combination, one object per
                  candidate
generate_batch()  same candidates and costs (bitwise), computed as arrays, for
                  LOW_SPEED + VELOCITY_KEEPING (the planner default); objects
                  are built on demand
"""

import copy
import numpy as np
from enum import Enum


# ==================================================
# 1. Planning modes
# ==================================================

class LateralMovement(Enum):
    HIGH_SPEED = "high_speed"
    LOW_SPEED = "low_speed"


class LongitudinalMovement(Enum):
    VELOCITY_KEEPING = "velocity_keeping"
    MERGING_AND_STOPPING = "merging_and_stopping"


# ==================================================
# 2. Frenet trajectory data
# ==================================================

class FrenetTrajectory:
    """Store a single Frenet trajectory candidate."""

    def __init__(self):
        self.t = np.array([])

        # Longitudinal derivatives: with respect to time
        self.s = np.array([])
        self.s_d = np.array([])
        self.s_dd = np.array([])
        self.s_ddd = np.array([])

        # LOW_SPEED: spatial derivatives of d(s)
        # HIGH_SPEED: converted spatial derivatives after generation
        self.d = np.array([])
        self.d_d = np.array([])
        self.d_dd = np.array([])
        self.d_ddd = np.array([])

        self.cf = float("inf")
        self.target_speed = 0.0
        self.target_d = 0.0
        self.target_s = None
        self.duration = 0.0


# ==================================================
# 3. Polynomial functions
# ==================================================

def _solve_quartic(s0, v0, a0, v1, a1, T):
    """Quartic s(t): initial s, v, a and terminal v, a."""

    if T <= 1e-8:
        raise ValueError("The polynomial duration must be positive.")

    c0 = s0
    c1 = v0
    c2 = a0 / 2.0

    A = np.array([
        [3.0 * T**2, 4.0 * T**3],
        [6.0 * T, 12.0 * T**2],
    ])

    b = np.array([
        v1 - c1 - 2.0 * c2 * T,
        a1 - 2.0 * c2,
    ])

    c3, c4 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4])


def _eval_quartic(c, t):
    """Evaluate position and the first three derivatives."""

    c0, c1, c2, c3, c4 = c

    position = c0 + c1*t + c2*t**2 + c3*t**3 + c4*t**4
    velocity = c1 + 2*c2*t + 3*c3*t**2 + 4*c4*t**3
    acceleration = 2*c2 + 6*c3*t + 12*c4*t**2
    jerk = 6*c3 + 24*c4*t

    return position, velocity, acceleration, jerk


def _solve_quintic(p0, v0, a0, p1, v1, a1, T):
    """Quintic polynomial with position, velocity and acceleration boundaries."""

    if T <= 1e-8:
        raise ValueError("The quintic polynomial interval must be positive.")

    c0 = p0
    c1 = v0
    c2 = a0 / 2.0

    A = np.array([
        [T**3, T**4, T**5],
        [3*T**2, 4*T**3, 5*T**4],
        [6*T, 12*T**2, 20*T**3],
    ])

    b = np.array([
        p1 - c0 - c1*T - c2*T**2,
        v1 - c1 - 2*c2*T,
        a1 - 2*c2,
    ])

    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([c0, c1, c2, c3, c4, c5])


def _eval_quintic(c, q):
    """Evaluate position and the first three derivatives."""

    c0, c1, c2, c3, c4, c5 = c

    position = (
        c0 + c1*q + c2*q**2 + c3*q**3 + c4*q**4 + c5*q**5
    )
    first = (
        c1 + 2*c2*q + 3*c3*q**2 + 4*c4*q**3 + 5*c5*q**4
    )
    second = (
        2*c2 + 6*c3*q + 12*c4*q**2 + 20*c5*q**3
    )
    third = 6*c3 + 24*c4*q + 60*c5*q**2

    return position, first, second, third


def _make_time_samples(T, dt):
    """Generate samples from zero and include terminal time exactly."""

    t = np.arange(0.0, T, dt)

    if len(t) == 0 or not np.isclose(t[-1], T):
        t = np.append(t, T)

    return t


def _make_range(start, stop, step):
    """Generate values including stop when necessary."""

    if step <= 0:
        raise ValueError("Sampling step must be positive.")

    values = np.arange(start, stop + step * 0.5, step)

    if len(values) == 0:
        return np.array([stop], dtype=float)

    return values


class CandidateSet:
    """
    Candidates of one generate_batch() call, stored as arrays.

    Candidate order (identical to generate()): duration, then terminal speed,
    then lateral offset. A FrenetTrajectory is only built on demand with
    path(index), so thousands of candidates never become objects.
    """

    def __init__(self, groups):
        # group: dict with t (n,), T, v_targets (m,), s/s_d/s_dd/s_ddd (m, n),
        #        d/d_d/d_dd/d_ddd (m, k, n), cf (m, k), offsets (k,)
        self.groups = groups
        sizes = [g["cf"].size for g in groups]
        self._starts = np.concatenate(([0], np.cumsum(sizes))).astype(int)
        self.cf = (
            np.concatenate([g["cf"].ravel() for g in groups])
            if groups else np.array([])
        )

    def __len__(self):
        return int(self.cf.size)

    def order(self):
        """Candidate indices by ascending cost (ties keep generation order)."""
        return np.argsort(self.cf, kind="stable")

    def path(self, index):
        g_i = int(np.searchsorted(self._starts, index, side="right")) - 1
        g = self.groups[g_i]
        k = len(g["offsets"])
        i, j = divmod(int(index) - int(self._starts[g_i]), k)

        fp = FrenetTrajectory.__new__(FrenetTrajectory)
        fp.t = g["t"]
        fp.s = g["s"][i]
        fp.s_d = g["s_d"][i]
        fp.s_dd = g["s_dd"][i]
        fp.s_ddd = g["s_ddd"][i]
        fp.target_speed = float(g["v_targets"][i])
        fp.target_s = None
        fp.duration = float(g["T"])
        fp.d = g["d"][i, j]
        fp.d_d = g["d_d"][i, j]
        fp.d_dd = g["d_dd"][i, j]
        fp.d_ddd = g["d_ddd"][i, j]
        fp.target_d = float(g["offsets"][j])
        fp.cf = float(g["cf"][i, j])
        return fp


# ==================================================
# 4. Frenet trajectory generator
# ==================================================

class FrenetTrajectoryGenerator:
    """
    Generate Frenet trajectory candidates.

    Inputs:
        s_state = [s, s_dot, s_ddot]
        d_state = [d, d_prime, d_prime_prime]

    LOW_SPEED:
        Generates d(s) directly.

    HIGH_SPEED:
        Generates d(t) and converts its derivatives to spatial derivatives.

    Returns:
        best_path, candidate_paths
    """

    def __init__(
        self,
        lateral_mode=LateralMovement.HIGH_SPEED,
        longitudinal_mode=LongitudinalMovement.VELOCITY_KEEPING,
        dt=0.1,
        min_t=0.5,
        max_t=1.5,
        target_speed=5.0,
        target_speed_step=0.5,
        n_speed_samples=1,
        stop_s=25.0,
        stop_s_step=2.0,
        n_stop_s_samples=2,
        lateral_offsets=(-1.0, 0.0, 1.0),
        min_s_dot_for_high_speed=0.1,
        k_j=1.0,
        k_t=0.1,
        k_speed=1.0,
        k_s=1.0,
        k_d=1.0,
        k_lat=1.0,
        k_lon=1.0,
    ):
        if dt <= 0 or min_t <= 0 or max_t < min_t:
            raise ValueError("Check dt, min_t, and max_t.")

        if target_speed < 0:
            raise ValueError("target_speed must be non-negative.")

        if target_speed_step <= 0:
            raise ValueError("target_speed_step must be positive.")

        if n_speed_samples < 0 or n_stop_s_samples < 0:
            raise ValueError("Sample counts must be non-negative.")

        if stop_s_step <= 0:
            raise ValueError("stop_s_step must be positive.")

        if min_s_dot_for_high_speed <= 0:
            raise ValueError(
                "min_s_dot_for_high_speed must be positive."
            )

        self.lateral_mode = self._parse_mode(
            lateral_mode, LateralMovement
        )
        self.longitudinal_mode = self._parse_mode(
            longitudinal_mode, LongitudinalMovement
        )

        self.dt = float(dt)
        self.min_t = float(min_t)
        self.max_t = float(max_t)

        self.target_speed = float(target_speed)
        self.target_speed_step = float(target_speed_step)
        self.n_speed_samples = int(n_speed_samples)

        self.stop_s = float(stop_s)
        self.stop_s_step = float(stop_s_step)
        self.n_stop_s_samples = int(n_stop_s_samples)

        self.lateral_offsets = tuple(
            float(d) for d in lateral_offsets
        )

        if not self.lateral_offsets:
            raise ValueError("At least one lateral offset is required.")

        self.min_s_dot_for_high_speed = float(
            min_s_dot_for_high_speed
        )

        self.k_j = float(k_j)
        self.k_t = float(k_t)
        self.k_speed = float(k_speed)
        self.k_s = float(k_s)
        self.k_d = float(k_d)
        self.k_lat = float(k_lat)
        self.k_lon = float(k_lon)

    @staticmethod
    def _parse_mode(mode, enum_type):
        """Accept enum members or their string values."""

        if isinstance(mode, enum_type):
            return mode

        if isinstance(mode, str):
            normalized = mode.strip().lower()

            for member in enum_type:
                if normalized in (
                    member.value.lower(),
                    member.name.lower(),
                ):
                    return member

        valid = ", ".join(member.value for member in enum_type)

        raise ValueError(
            f"Invalid {enum_type.__name__}: {mode}. "
            f"Choose one of: {valid}"
        )

    def _get_speed_targets(self):
        """Generate terminal speed candidates."""

        offsets = np.arange(
            -self.n_speed_samples,
            self.n_speed_samples + 1,
        ) * self.target_speed_step

        targets = self.target_speed + offsets
        targets = targets[targets >= 0.0]

        return np.unique(np.append(targets, self.target_speed))

    def _get_stop_positions(self, s0):
        """Generate terminal positions around stop_s."""

        if s0 >= self.stop_s:
            return np.array([], dtype=float)

        offsets = np.arange(
            -self.n_stop_s_samples,
            self.n_stop_s_samples + 1,
        ) * self.stop_s_step

        targets = self.stop_s + offsets
        targets = targets[targets > s0 + 1e-6]

        return np.unique(targets)

    def _generate_longitudinal_candidates(self, s_state, durations):
        """Generate longitudinal trajectories."""

        s0, s_dot0, s_ddot0 = s_state
        candidates = []

        for T in durations:
            t = _make_time_samples(T, self.dt)

            if (
                self.longitudinal_mode
                == LongitudinalMovement.VELOCITY_KEEPING
            ):
                destinations = [
                    (target_v, None)
                    for target_v in self._get_speed_targets()
                ]

                for target_v, target_s in destinations:
                    coeff = _solve_quartic(
                        s0, s_dot0, s_ddot0,
                        target_v, 0.0, T,
                    )

                    s, s_d, s_dd, s_ddd = _eval_quartic(coeff, t)

                    if np.any(s_d < -1e-6):
                        continue

                    if s[-1] - s[0] <= 1e-6:
                        continue

                    fp = FrenetTrajectory()
                    fp.t = t.copy()
                    fp.s = s
                    fp.s_d = s_d
                    fp.s_dd = s_dd
                    fp.s_ddd = s_ddd
                    fp.target_speed = float(target_v)
                    fp.target_s = None
                    fp.duration = float(T)

                    candidates.append(fp)

            else:
                for target_s in self._get_stop_positions(s0):
                    coeff = _solve_quintic(
                        s0, s_dot0, s_ddot0,
                        target_s, 0.0, 0.0, T,
                    )

                    s, s_d, s_dd, s_ddd = _eval_quintic(coeff, t)

                    if np.any(s_d < -1e-6):
                        continue

                    if np.any(s < s0 - 1e-6):
                        continue

                    fp = FrenetTrajectory()
                    fp.t = t.copy()
                    fp.s = s
                    fp.s_d = s_d
                    fp.s_dd = s_dd
                    fp.s_ddd = s_ddd
                    fp.target_speed = 0.0
                    fp.target_s = float(target_s)
                    fp.duration = float(T)

                    candidates.append(fp)

        return candidates

    def _generate_lateral_trajectory(self, fp, d_state, target_d):
        """Generate lateral trajectory and return its jerk for cost."""

        d0, d_prime0, d_prime_prime0 = d_state
        s0 = fp.s[0]
        distance = fp.s[-1] - s0

        if distance <= 1e-6:
            return None, None

        if self.lateral_mode == LateralMovement.LOW_SPEED:
            # d = d(s): all derivatives are spatial derivatives.
            coeff = _solve_quintic(
                d0, d_prime0, d_prime_prime0,
                target_d, 0.0, 0.0, distance,
            )

            q = fp.s - s0
            d, d_d, d_dd, d_ddd = _eval_quintic(coeff, q)

            return (d, d_d, d_dd, d_ddd), d_ddd

        # d = d(t): convert initial spatial derivatives to time derivatives.
        if np.any(fp.s_d <= self.min_s_dot_for_high_speed):
            return None, None

        d_dot0 = d_prime0 * fp.s_d[0]
        d_ddot0 = (
            d_prime_prime0 * fp.s_d[0]**2
            + d_prime0 * fp.s_dd[0]
        )

        coeff = _solve_quintic(
            d0, d_dot0, d_ddot0,
            target_d, 0.0, 0.0, fp.duration,
        )

        d, d_dot, d_ddot, d_jerk_t = _eval_quintic(coeff, fp.t)

        # Convert time derivatives to spatial derivatives.
        d_d = d_dot / fp.s_d

        d_dd = (
            d_ddot - d_d * fp.s_dd
        ) / (fp.s_d**2)

        d_ddd = (
            d_jerk_t
            - 3.0 * d_dd * fp.s_d * fp.s_dd
            - d_d * fp.s_ddd
        ) / (fp.s_d**3)

        return (d, d_d, d_dd, d_ddd), d_jerk_t

    def _low_speed_candidates(self, lon_path, d_state):
        """
        LOW_SPEED lateral candidates of one longitudinal path for ALL lateral
        offsets at once. Same maths as _generate_lateral_trajectory plus the
        cost in generate(); the longitudinal arrays are shared, not copied.
        """
        d0, d_prime0, d_prime_prime0 = d_state
        s0 = lon_path.s[0]
        distance = lon_path.s[-1] - s0

        if distance <= 1e-6:
            return []

        if distance <= 1e-8:
            raise ValueError("The quintic polynomial interval must be positive.")

        T = distance
        targets = np.asarray(self.lateral_offsets, dtype=float)
        k = len(targets)

        c0 = d0
        c1 = d_prime0
        c2 = d_prime_prime0 / 2.0

        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3],
        ])
        b = np.empty((3, k))
        b[0] = targets - c0 - c1*T - c2*T**2
        b[1] = 0.0 - c1 - 2*c2*T
        b[2] = 0.0 - 2*c2

        c345 = np.linalg.solve(A, b)           # (3, k)
        c3 = c345[0][:, None]
        c4 = c345[1][:, None]
        c5 = c345[2][:, None]

        q = (lon_path.s - s0)[None, :]
        d = c0 + c1*q + c2*q**2 + c3*q**3 + c4*q**4 + c5*q**5
        d_d = c1 + 2*c2*q + 3*c3*q**2 + 4*c4*q**3 + 5*c5*q**4
        d_dd = 2*c2 + 6*c3*q + 12*c4*q**2 + 20*c5*q**3
        d_ddd = 6*c3 + 24*c4*q + 60*c5*q**2

        J_lat = (
            self.k_j * np.sum(d_ddd**2, axis=1) * self.dt
            + self.k_t * lon_path.duration
            + self.k_d * d[:, -1]**2
        )

        if self.longitudinal_mode == LongitudinalMovement.VELOCITY_KEEPING:
            destination_cost = self.k_speed * (self.target_speed - lon_path.s_d[-1])**2
        else:
            destination_cost = self.k_s * (lon_path.target_s - lon_path.s[-1])**2

        J_lon = (
            self.k_j * np.sum(lon_path.s_ddd**2) * self.dt
            + self.k_t * lon_path.duration
            + destination_cost
        )

        cf = self.k_lat * J_lat + self.k_lon * J_lon

        out = []
        for i in range(k):
            # every attribute is set below, so skip the empty-array __init__
            fp = FrenetTrajectory.__new__(FrenetTrajectory)
            fp.t = lon_path.t
            fp.s = lon_path.s
            fp.s_d = lon_path.s_d
            fp.s_dd = lon_path.s_dd
            fp.s_ddd = lon_path.s_ddd
            fp.target_speed = lon_path.target_speed
            fp.target_s = lon_path.target_s
            fp.duration = lon_path.duration
            fp.d = d[i]
            fp.d_d = d_d[i]
            fp.d_dd = d_dd[i]
            fp.d_ddd = d_ddd[i]
            fp.target_d = float(targets[i])
            fp.cf = float(cf[i])
            out.append(fp)

        return out

    @property
    def supports_batch(self):
        return (
            self.lateral_mode == LateralMovement.LOW_SPEED
            and self.longitudinal_mode == LongitudinalMovement.VELOCITY_KEEPING
        )

    def _check_states(self, s_state, d_state):
        s_state = np.asarray(s_state, dtype=float)
        d_state = np.asarray(d_state, dtype=float)

        if s_state.shape != (3,) or d_state.shape != (3,):
            raise ValueError(
                "s_state and d_state must each contain three values."
            )

        if (
            not np.all(np.isfinite(s_state))
            or not np.all(np.isfinite(d_state))
        ):
            raise ValueError("Frenet states must contain finite values.")

        if s_state[1] < 0:
            raise ValueError(
                "This implementation assumes forward motion."
            )

        return s_state, d_state

    def generate_batch(self, s_state, d_state):
        """
        Same candidates and costs as generate() for LOW_SPEED + VELOCITY_KEEPING,
        computed for all terminal speeds and lateral offsets of a duration at
        once. Returns a CandidateSet (empty if no candidate exists).
        """
        if not self.supports_batch:
            raise ValueError("generate_batch needs LOW_SPEED and VELOCITY_KEEPING.")

        s_state, d_state = self._check_states(s_state, d_state)
        s0, s_dot0, s_ddot0 = s_state
        d0, d_prime0, d_prime_prime0 = d_state

        offsets = np.asarray(self.lateral_offsets, dtype=float)
        k = len(offsets)
        v_targets_all = self._get_speed_targets()
        groups = []

        for T in _make_range(self.min_t, self.max_t, self.dt):
            if T <= 1e-8:
                raise ValueError("The polynomial duration must be positive.")

            t = _make_time_samples(T, self.dt)
            m = len(v_targets_all)

            # ---- longitudinal quartic for every terminal speed ----------
            c0, c1, c2 = s0, s_dot0, s_ddot0 / 2.0
            A = np.array([
                [3.0 * T**2, 4.0 * T**3],
                [6.0 * T, 12.0 * T**2],
            ])
            b = np.empty((m, 2, 1))
            b[:, 0, 0] = v_targets_all - c1 - 2.0 * c2 * T
            b[:, 1, 0] = 0.0 - 2.0 * c2
            c34 = np.linalg.solve(np.broadcast_to(A, (m, 2, 2)), b)[:, :, 0]
            c3 = c34[:, 0][:, None]
            c4 = c34[:, 1][:, None]

            tt = t[None, :]
            s = c0 + c1*tt + c2*tt**2 + c3*tt**3 + c4*tt**4
            s_d = c1 + 2*c2*tt + 3*c3*tt**2 + 4*c4*tt**3
            s_dd = 2*c2 + 6*c3*tt + 12*c4*tt**2
            s_ddd = 6*c3 + 24*c4*tt

            keep = ~np.any(s_d < -1e-6, axis=1) & ((s[:, -1] - s[:, 0]) > 1e-6)
            if not keep.any():
                continue

            v_targets = v_targets_all[keep]
            s, s_d, s_dd, s_ddd = s[keep], s_d[keep], s_dd[keep], s_ddd[keep]
            n_keep = len(v_targets)

            # ---- lateral quintic d(s) for every (speed, offset) ----------
            dist = s[:, -1] - s[:, 0]
            if np.any(dist <= 1e-8):
                raise ValueError("The quintic polynomial interval must be positive.")

            # Built per path with scalar arithmetic: scalar pow can differ from
            # array pow in the last bit, and the costs must match generate().
            Aq = np.empty((n_keep, 3, 3))
            bq = np.empty((n_keep, 3, k))
            e0, e1, e2 = d0, d_prime0, d_prime_prime0 / 2.0

            for r, D in enumerate(dist):
                Aq[r] = [
                    [D**3, D**4, D**5],
                    [3*D**2, 4*D**3, 5*D**4],
                    [6*D, 12*D**2, 20*D**3],
                ]
                bq[r, 0] = offsets - e0 - e1*D - e2*D**2
                bq[r, 1] = 0.0 - e1 - 2*e2*D
                bq[r, 2] = 0.0 - 2*e2

            c345 = np.linalg.solve(Aq, bq)                   # (m, 3, k)
            q3 = c345[:, 0, :, None]
            q4 = c345[:, 1, :, None]
            q5 = c345[:, 2, :, None]

            q = (s - s[:, :1])[:, None, :]                   # (m, 1, n)
            d = e0 + e1*q + e2*q**2 + q3*q**3 + q4*q**4 + q5*q**5
            d_d = e1 + 2*e2*q + 3*q3*q**2 + 4*q4*q**3 + 5*q5*q**4
            d_dd = 2*e2 + 6*q3*q + 12*q4*q**2 + 20*q5*q**3
            d_ddd = 6*q3 + 24*q4*q + 60*q5*q**2

            J_lat = (
                self.k_j * np.sum(d_ddd**2, axis=2) * self.dt
                + self.k_t * float(T)
                + self.k_d * d[:, :, -1]**2
            )
            # per path with np.sum / scalar pow, exactly as generate() does
            J_lon = np.array([
                self.k_j * np.sum(s_ddd[r]**2) * self.dt
                + self.k_t * float(T)
                + self.k_speed * (self.target_speed - s_d[r, -1])**2
                for r in range(n_keep)
            ])
            cf = self.k_lat * J_lat + self.k_lon * J_lon[:, None]

            groups.append({
                "t": t, "T": T, "v_targets": v_targets, "offsets": offsets,
                "s": s, "s_d": s_d, "s_dd": s_dd, "s_ddd": s_ddd,
                "d": d, "d_d": d_d, "d_dd": d_dd, "d_ddd": d_ddd, "cf": cf,
            })

        return CandidateSet(groups)

    def generate(self, s_state, d_state):
        """Generate candidates and return the minimum-cost candidate."""

        s_state = np.asarray(s_state, dtype=float)
        d_state = np.asarray(d_state, dtype=float)

        if s_state.shape != (3,) or d_state.shape != (3,):
            raise ValueError(
                "s_state and d_state must each contain three values."
            )

        if (
            not np.all(np.isfinite(s_state))
            or not np.all(np.isfinite(d_state))
        ):
            raise ValueError("Frenet states must contain finite values.")

        s0, s_dot0, s_ddot0 = s_state

        if s_dot0 < 0:
            raise ValueError(
                "This implementation assumes forward motion."
            )

        durations = _make_range(
            self.min_t, self.max_t, self.dt
        )

        lon_candidates = self._generate_longitudinal_candidates(
            s_state, durations
        )

        complete_candidates = []

        for lon_path in lon_candidates:
            if self.lateral_mode == LateralMovement.LOW_SPEED:
                complete_candidates.extend(
                    self._low_speed_candidates(lon_path, d_state)
                )
                continue

            for target_d in self.lateral_offsets:
                lateral_result, jerk_for_cost = (
                    self._generate_lateral_trajectory(
                        lon_path, d_state, target_d
                    )
                )

                if lateral_result is None:
                    continue

                d, d_d, d_dd, d_ddd = lateral_result

                # FIX: Each lateral candidate must own its own object.
                fp = copy.deepcopy(lon_path)

                fp.d = np.asarray(d, dtype=float).copy()
                fp.d_d = np.asarray(d_d, dtype=float).copy()
                fp.d_dd = np.asarray(d_dd, dtype=float).copy()
                fp.d_ddd = np.asarray(d_ddd, dtype=float).copy()
                fp.target_d = float(target_d)

                J_lat = (
                    self.k_j * np.sum(np.asarray(jerk_for_cost)**2) * self.dt
                    + self.k_t * fp.duration
                    + self.k_d * fp.d[-1]**2
                )

                if (
                    self.longitudinal_mode
                    == LongitudinalMovement.VELOCITY_KEEPING
                ):
                    destination_cost = (
                        self.k_speed
                        * (self.target_speed - fp.s_d[-1])**2
                    )
                else:
                    destination_cost = (
                        self.k_s * (fp.target_s - fp.s[-1])**2
                    )

                J_lon = (
                    self.k_j * np.sum(fp.s_ddd**2) * self.dt
                    + self.k_t * fp.duration
                    + destination_cost
                )

                fp.cf = (
                    self.k_lat * J_lat
                    + self.k_lon * J_lon
                )

                complete_candidates.append(fp)

        if not complete_candidates:
            raise RuntimeError(
                "No valid Frenet candidates were generated. "
                "Check the selected modes, initial state, stop position, "
                "lateral offsets, and time horizon."
            )

        best_path = min(
            complete_candidates,
            key=lambda path: path.cf,
        )

        return best_path, complete_candidates