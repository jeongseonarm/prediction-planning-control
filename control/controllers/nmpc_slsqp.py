import numpy as np
from scipy.optimize import minimize


class MPCController:

    def __init__(self, model, horizon=10):
        self.model = model
        self.N = horizon  # Prediction horizon

        # ==================================
        # State cost matrix Q: [x, y, theta, v]
        # ==================================
        self.Q = np.diag([10.0, 10.0, 5.0, 2.0])
        self.Qf = self.Q  # Terminal state cost matrix

        # ==================================
        # Input cost matrix R: [a, delta]
        # ==================================
        self.R = np.diag([0.1, 1.0])

        # ==================================
        # Input rate cost matrix R_delta: [da, ddelta]
        # ==================================
        self.R_delta = np.diag([1.0, 10.0])

        # ==================================
        # Control history for Warm Start & Rate cost
        # ==================================
        self.previous_control = None  # u_{k-1}
        self.previous_controls_sequence = None  # Full horizon sequence

        # ==================================
        # Input limits [a, delta]
        # ==================================
        self.a_min = -1.0  # [m/s^2]
        self.a_max = 1.0  # [m/s^2]

        self.delta_min = np.deg2rad(-30)  # [rad]
        self.delta_max = np.deg2rad(30)  # [rad]

        # fast cost evaluation (see _fast_setup); None = not checked yet
        self._fast = None
        self._fast_key = None

    def predict(self, state, controls):
        """Predicts N+1 future states given initial state and N control inputs."""
        # models may provide a faster rollout with the same arithmetic
        if hasattr(self.model, "rollout"):
            return self.model.rollout(state, controls)

        states = [state.copy()]
        x = state.copy()

        for u in controls:
            x = self.model.update(x, u)
            states.append(x.copy())

        return np.array(states)  # Shape: (N+1, 4)

    # ------------------------------------------------------------------
    # fast cost evaluation
    #
    # Same value as _cost_function_reference, bit for bit, but without a
    # Python loop of small matrix products. It relies on Q, Qf, R, R_delta
    # being diagonal (then x^T D x = sum_j (x_j * D_jj) * x_j exactly) and is
    # only used after a self check against the reference implementation.
    # ------------------------------------------------------------------
    def _fast_setup(self):
        mats = (self.Q, self.Qf, self.R, self.R_delta)
        key = tuple(np.asarray(m, dtype=float).tobytes() for m in mats) + (self.N,)
        if self._fast is not None and key == self._fast_key:
            return self._fast

        self._fast_key = key
        self._fast = None

        def diagonal(m):
            m = np.asarray(m, dtype=float)
            return m.shape == (len(m), len(m)) and not np.any(m - np.diag(np.diag(m)))

        if not all(diagonal(m) for m in mats):
            return None

        q = np.diag(np.asarray(self.Q, dtype=float))
        qf = np.diag(np.asarray(self.Qf, dtype=float))
        params = {
            "q": np.vstack([np.tile(q, (self.N, 1)), qf[None, :]]),   # (N+1, 4)
            "r": np.diag(np.asarray(self.R, dtype=float)),
            "rd": np.diag(np.asarray(self.R_delta, dtype=float)),
        }

        # self check on random inputs: both implementations must agree exactly
        rng = np.random.default_rng(0)
        self._fast = params
        try:
            for _ in range(200):
                state = np.array([
                    rng.uniform(0, 100), rng.uniform(-80, 0),
                    rng.uniform(-3.1, 3.1), rng.uniform(0, 8),
                ])
                ref = state[None, :] + rng.normal(0, 1.0, (self.N + 1, 4))
                u = np.c_[rng.uniform(-3, 3, self.N), rng.uniform(-0.5, 0.5, self.N)]
                saved = self.previous_control
                self.previous_control = u[0] + rng.normal(0, 0.1, 2) if _ % 2 else None
                try:
                    a = self._cost_function_fast(u.ravel(), state, ref, params)
                    b = self._cost_function_reference(u.ravel(), state, ref)
                finally:
                    self.previous_control = saved
                if not (a == b):
                    self._fast = None
                    break
        except Exception:
            self._fast = None
        return self._fast

    def _cost_function_fast(self, u_flat, state, reference, params):
        N = self.N
        controls = u_flat.reshape(N, 2)
        predicted_states = self.predict(state, controls)

        error = predicted_states - reference[: N + 1]
        if not np.all(np.isfinite(error)):
            return self._cost_function_reference(u_flat, state, reference)
        error[:, 2] = np.arctan2(np.sin(error[:, 2]), np.cos(error[:, 2]))

        weighted_error = error * params["q"]
        weighted_u = controls * params["r"]

        du = np.empty_like(controls)
        if self.previous_control is None:
            du[0] = 0.0
        else:
            du[0] = controls[0] - self.previous_control
        du[1:] = controls[1:] - controls[:-1]
        weighted_du = du * params["rd"]

        dot = np.dot
        cost = 0.0
        for k in range(N + 1):
            cost += dot(weighted_error[k], error[k])
        for k in range(N):
            cost += dot(weighted_u[k], controls[k])
            cost += dot(weighted_du[k], du[k])

        return cost

    def cost_function(self, u_flat, state, reference):
        """MPC Objective Function.

        reference shape: (N+1, 4) -> [x, y, theta, v] for each step k=0...N
        """
        params = self._fast_setup()
        if params is not None:
            return self._cost_function_fast(u_flat, state, reference, params)
        return self._cost_function_reference(u_flat, state, reference)

    def _cost_function_reference(self, u_flat, state, reference):
        """Original implementation, kept as the reference and as the fallback."""
        controls = u_flat.reshape(self.N, 2)
        predicted_states = self.predict(state, controls)

        cost = 0.0

        # 1. State Tracking Cost (k = 0 ... N) -> N+1 points
        for k in range(self.N + 1):
            error = predicted_states[k] - reference[k]

            # Normalize heading angle error (Index 2: theta)
            error[2] = np.arctan2(np.sin(error[2]), np.cos(error[2]))

            Q_mat = self.Qf if k == self.N else self.Q
            cost += error.T @ Q_mat @ error

        # 2. Input & Input Rate Cost (k = 0 ... N-1) -> N points
        for k in range(self.N):
            u = controls[k]

            # Input Magnitude Cost
            cost += u.T @ self.R @ u

            # Input Rate Cost
            if k == 0:
                if self.previous_control is None:
                    du = np.zeros(2)
                else:
                    du = u - self.previous_control
            else:
                du = u - controls[k - 1]

            cost += du.T @ self.R_delta @ du

        return cost

    def compute_control(self, state, reference):
        """Computes optimal control input.

        state: current state [x, y, theta, v]
        reference: target trajectory matrix of shape (N+1, 4)
        """

        # ==================================
        # Warm Start Initialization
        # ==================================
        if self.previous_controls_sequence is None:
            u0 = np.zeros(self.N * 2)
        else:
            # Shift previous sequence by 1 step
            u_shift = np.vstack(
                (
                    self.previous_controls_sequence[1:],
                    self.previous_controls_sequence[-1],
                )
            )
            u0 = u_shift.flatten()

        # ==================================
        # Input Bounds: [a, delta]
        # ==================================
        bounds = [
            (self.a_min, self.a_max),
            (self.delta_min, self.delta_max),
        ] * self.N

        # ==================================
        # Optimization 
        # ==================================
        result = minimize(
            self.cost_function,
            u0,
            args=(state, reference),
            bounds=bounds,
            method="SLSQP",
            options={"maxiter": 50, "ftol": 1e-4},
        )

        if not result.success:
            print(f"[MPC Warning] Optimization failed: {result.message}")

        optimal_controls = result.x.reshape(self.N, 2)

        # ==================================
        # Receding Horizon Update
        # ==================================
        control = optimal_controls[0]

        self.previous_control = control.copy()
        self.previous_controls_sequence = optimal_controls.copy()

        return control