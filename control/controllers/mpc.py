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

    def predict(self, state, controls):
        """Predicts N+1 future states given initial state and N control inputs."""
        states = [state.copy()]
        x = state.copy()

        for u in controls:
            x = self.model.update(x, u)
            states.append(x.copy())

        return np.array(states)  # Shape: (N+1, 4)

    def cost_function(self, u_flat, state, reference):
        """MPC Objective Function.

        reference shape: (N+1, 4) -> [x, y, theta, v] for each step k=0...N
        """
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