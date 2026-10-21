import numpy as np
import osqp
from scipy import sparse


class LTVMPCController:
    """
    Linear time-varying MPC solved with OSQP (one QP per call).

    The nonlinear model is linearised around a nominal rollout (the previous
    solution shifted by one step), so the QP is convex and solves in a few ms.

    state     [x, y, theta, v]
    reference (horizon + 1, 4) rows of [x, y, theta, v]
    control   [a, delta]

    first_step_rate_penalty
        True  (default): without a previous control the first step is
              penalised as a change from zero, like the original code.
        False: the first step is free of the rate cost until a control exists.
    update_history_on_failure
        False (default): when OSQP fails the nominal control is returned but
              the control history is left unchanged, like the original code.
        True: the fallback control also becomes the history.
    """

    def __init__(
        self,
        model,
        horizon=10,
        first_step_rate_penalty=True,
        update_history_on_failure=False,
    ):
        self.model = model
        self.N = horizon  # Prediction horizon
        self.first_step_rate_penalty = first_step_rate_penalty
        self.update_history_on_failure = update_history_on_failure

        # State cost matrix Q: [x, y, theta, v]
        self.Q = np.diag([10.0, 10.0, 5.0, 2.0])
        self.Qf = self.Q.copy()

        # Input cost matrix R: [a, delta]
        self.R = np.diag([0.1, 1.0])

        # Input rate cost matrix R_delta
        self.R_delta = np.diag([1.0, 10.0])

        # Control history
        self.previous_control = None
        self.previous_controls_sequence = None

        # Input limits [a, delta]
        self.a_min = -1.0
        self.a_max = 1.0

        self.delta_min = np.deg2rad(-30)
        self.delta_max = np.deg2rad(30)

    def reset(self):
        """Forget the control history (new scenario)."""
        self.previous_control = None
        self.previous_controls_sequence = None

    @staticmethod
    def _angle_error(angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def _prepare_reference(self, state, reference):
        """Keep reference headings on a continuous angle branch."""
        reference = np.asarray(reference, dtype=float).copy()

        reference[0, 2] = state[2] + self._angle_error(reference[0, 2] - state[2])

        for k in range(1, self.N + 1):
            reference[k, 2] = reference[k - 1, 2] + self._angle_error(
                reference[k, 2] - reference[k - 1, 2]
            )

        return reference

    def nominal_controls(self, reference):
        """Estimate initial nominal controls from reference if no past control sequence exists."""
        controls = np.zeros((self.N, 2))
        dt = self.model.dt
        L = self.model.L

        for k in range(self.N):
            v = reference[k, 3]
            v_next = reference[k + 1, 3]

            controls[k, 0] = (v_next - v) / dt

            dtheta = self._angle_error(
                reference[k + 1, 2] - reference[k, 2]
            )

            if abs(v) > 1e-3:
                controls[k, 1] = np.arctan(L * dtheta / (v * dt))

        controls[:, 0] = np.clip(controls[:, 0], self.a_min, self.a_max)
        controls[:, 1] = np.clip(controls[:, 1], self.delta_min, self.delta_max)

        return controls

    def _generate_nominal_trajectory(self, state, reference):
        """Rollout nonlinear vehicle model from current state x0."""
        if self.previous_controls_sequence is None:
            u_nominal = self.nominal_controls(reference)
        else:
            u_nominal = np.vstack((
                self.previous_controls_sequence[1:],
                self.previous_controls_sequence[-1],
            ))

        x_nominal = np.zeros((self.N + 1, 4))
        x_nominal[0] = state.copy()

        for k in range(self.N):
            x_next = self.model.update(x_nominal[k], u_nominal[k])
            x_next[2] = reference[k + 1, 2] + self._angle_error(x_next[2] - reference[k + 1, 2])
            x_nominal[k + 1] = x_next

        return x_nominal, u_nominal

    def _build_qp(self, state, reference, x_nominal, u_nominal):
        """Build the OSQP problem using the nominal rollout trajectory."""
        N = self.N
        nx = 4
        nu = 2

        n_x = (N + 1) * nx
        n_u = N * nu
        n_var = n_x + n_u

        def x_slice(k):
            return slice(k * nx, (k + 1) * nx)

        def u_slice(k):
            start = n_x + k * nu
            return slice(start, start + nu)

        # 1. Objective Matrices (P, q)
        P = np.zeros((n_var, n_var))
        q = np.zeros(n_var)

        for k in range(N + 1):
            Q_mat = self.Qf if k == N else self.Q
            idx = x_slice(k)

            P[idx, idx] += 2.0 * Q_mat
            q[idx] += -2.0 * Q_mat @ reference[k]

        for k in range(N):
            idx = u_slice(k)
            P[idx, idx] += 2.0 * self.R

        D = np.zeros((N * nu, n_var))
        d = np.zeros(N * nu)

        for k in range(N):
            rows = slice(k * nu, (k + 1) * nu)
            D[rows, u_slice(k)] = np.eye(nu)

            if k == 0:
                if self.previous_control is not None:
                    d[rows] = -self.previous_control
            else:
                D[rows, u_slice(k - 1)] = -np.eye(nu)

        if self.previous_control is None and not self.first_step_rate_penalty:
            D[0:nu, :] = 0.0

        R_bar = np.kron(np.eye(N), self.R_delta)
        P += 2.0 * D.T @ R_bar @ D
        q += 2.0 * D.T @ R_bar @ d

        # 2. Dynamics Constraints (A_qp, lower, upper)
        n_eq = (N + 1) * nx
        n_bounds = N * nu

        A_qp = np.zeros((n_eq + n_bounds, n_var))
        lower = np.zeros(n_eq + n_bounds)
        upper = np.zeros(n_eq + n_bounds)

        A_qp[:nx, x_slice(0)] = np.eye(nx)
        lower[:nx] = state
        upper[:nx] = state

        for k in range(N):
            rows = slice((k + 1) * nx, (k + 2) * nx)

            x_bar = x_nominal[k]
            u_bar = u_nominal[k]

            A_k, B_k = self.model.linearize(x_bar, u_bar)
            f_bar = self.model.update(x_bar, u_bar)

            f_bar[2] = reference[k + 1, 2] + self._angle_error(
                f_bar[2] - reference[k + 1, 2]
            )

            c_k = f_bar - A_k @ x_bar - B_k @ u_bar

            A_qp[rows, x_slice(k + 1)] = np.eye(nx)
            A_qp[rows, x_slice(k)] = -A_k
            A_qp[rows, u_slice(k)] = -B_k

            lower[rows] = c_k
            upper[rows] = c_k

        # 3. Input Bounds
        offset = n_eq
        for k in range(N):
            rows = slice(offset + k * nu, offset + (k + 1) * nu)
            A_qp[rows, u_slice(k)] = np.eye(nu)

            lower[rows] = [self.a_min, self.delta_min]
            upper[rows] = [self.a_max, self.delta_max]

        P = sparse.csc_matrix(np.triu(P))
        A_qp = sparse.csc_matrix(A_qp)

        return P, q, A_qp, lower, upper

    def compute_control(self, state, reference):
        """Compute optimal control using Nominal Rollout LTV MPC."""
        state = np.asarray(state, dtype=float)
        reference = np.asarray(reference, dtype=float)

        if state.shape != (4,):
            raise ValueError("state must have shape (4,)")

        if reference.shape != (self.N + 1, 4):
            raise ValueError(f"reference must have shape ({self.N + 1}, 4)")

        reference = self._prepare_reference(state, reference)

        x_nominal, u_nominal = self._generate_nominal_trajectory(state, reference)

        P, q, A, lower, upper = self._build_qp(
            state, reference, x_nominal, u_nominal
        )

        solver = osqp.OSQP()
        solver.setup(
            P=P,
            q=q,
            A=A,
            l=lower,
            u=upper,
            verbose=False,
            warm_starting=True,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=4000,
        )

        z0 = np.concatenate((x_nominal.flatten(), u_nominal.flatten()))
        solver.warm_start(x=z0)

        result = solver.solve()
        status = result.info.status.lower()

        if status not in ("solved", "solved inaccurate"):
            print(f"[MPC Warning] OSQP failed: {result.info.status}")
            control = u_nominal[0].copy()

            if self.update_history_on_failure:
                self.previous_control = control.copy()
                self.previous_controls_sequence = u_nominal.copy()

            return control

        z_opt = result.x
        optimal_controls = z_opt[(self.N + 1) * 4:].reshape(self.N, 2)

        control = optimal_controls[0].copy()
        self.previous_control = control.copy()
        self.previous_controls_sequence = optimal_controls.copy()

        return control