import numpy as np


class KinematicBicycle:
    """Kinematic Bicycle Model with Acceleration Input

    State:
        x     : position x [m]
        y     : position y [m]
        theta : heading angle [rad]
        v     : velocity [m/s]

    Control:
        a     : acceleration [m/s^2]
        delta : steering angle [rad]
    """

    def __init__(self, wheelbase, dt):
        self.L = wheelbase
        self.dt = dt

    def update(self, state, control):
        x, y, theta, v = state
        a, delta = control

        # continuous model 
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = (v / self.L) * np.tan(delta)
        v_dot = a

        # Euler integration
        x_next = x + x_dot * self.dt
        y_next = y + y_dot * self.dt
        theta_next = theta + theta_dot * self.dt
        v_next = v + v_dot * self.dt

        # normalize heading (-pi ~ pi)
        theta_next = (theta_next + np.pi) % (2 * np.pi) - np.pi

        return np.array([x_next, y_next, theta_next, v_next])