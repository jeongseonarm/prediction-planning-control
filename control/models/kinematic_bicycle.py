import numpy as np


class KinematicBicycle:
    """Kinematic Bicycle Model with Acceleration Input.

    State:   [x, y, theta, v]
    Control: [a, delta]
    """

    def __init__(self, wheelbase, dt):
        self.L = wheelbase
        self.dt = dt

    def update(self, state, control):
        """Nonlinear vehicle state update."""
        x, y, theta, v = state
        a, delta = control

        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = (v / self.L) * np.tan(delta)
        v_dot = a

        x_next = x + x_dot * self.dt
        y_next = y + y_dot * self.dt
        theta_next = theta + theta_dot * self.dt
        v_next = v + v_dot * self.dt

        theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_next, y_next, theta_next, v_next])

    def rollout(self, state, controls):
        """
        N + 1 states from `state` under `controls` (shape (N, 2)).
        Same arithmetic as calling update() N times, on Python floats so that
        no numpy array is created per step. Returns an (N + 1, 4) array.
        """
        x, y, theta, v = (float(s) for s in state)
        L = self.L
        dt = self.dt
        two_pi = 2 * np.pi
        rows = [(x, y, theta, v)]

        for a, delta in np.asarray(controls, dtype=float).tolist():
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta)
            theta_dot = (v / L) * np.tan(delta)

            x_next = x + x_dot * dt
            y_next = y + y_dot * dt
            theta_next = theta + theta_dot * dt
            v_next = v + a * dt

            theta_next = (theta_next + np.pi) % two_pi - np.pi

            x, y, theta, v = x_next, y_next, theta_next, v_next
            rows.append((x, y, theta, v))

        return np.array(rows, dtype=float)

    def linearize(self, state, control):
        """Linearize the discrete-time model around nominal state and control."""
        _, _, theta, v = state
        _, delta = control

        dt = self.dt
        L = self.L

        A = np.eye(4)
        A[0, 2] = -v * np.sin(theta) * dt
        A[0, 3] = np.cos(theta) * dt

        A[1, 2] = v * np.cos(theta) * dt
        A[1, 3] = np.sin(theta) * dt

        A[2, 3] = np.tan(delta) * dt / L

        B = np.zeros((4, 2))
        B[2, 1] = v * dt / (L * np.cos(delta) ** 2)
        B[3, 0] = dt

        return A, B