# import numpy as np


# class PurePursuitController:

#     def __init__(
#             self,
#             wheelbase,
#             velocity=10.0,
#             constant=0.5
#     ):
#         """
#         Pure Pursuit Controller

#         Parameters
#         ----------
#         wheelbase : float
#             차량 wheelbase [m]

#         velocity : float
#             Constant velocity [m/s]
#         """

#         self.L = wheelbase
#         self.v = velocity
#         self.k = constant

#         self.delta_min = np.deg2rad(-30)
#         self.delta_max = np.deg2rad(30)


#     def compute_control(
#             self,
#             state,
#             x_ref,
#             y_ref
#     ):
#         """
#         Parameters
#         ----------
#         state :
#             [x, y, theta]

#         x_ref :
#             reference x

#         y_ref :
#             reference y

#         Returns
#         -------
#         control :
#             [v, delta]
#         """

#         x = state[0]
#         y = state[1]
#         theta = state[2]

#         Ld = self.k * self.v

#         Px = x + Ld * np.cos(theta)
#         Py = y + Ld * np.sin(theta)

#         distance = np.sqrt(
#                             (x_ref - Px) ** 2 +
#                             (y_ref - Py) ** 2
#                             )

#         idx = np.argmin(distance)

#         L_bar = distance[idx]  
#         Cx = x_ref[idx]  # Close x        
#         Cy = y_ref[idx]  # Close y

#         # P1(x,y), P2(px,py), C(cx,cy)
#         a = y - Py
#         b = Px - x
#         c = x * Py - Px * y

#         y2 = (a * Cx + b * Cy + c) / np.sqrt(a**2 + b**2)

#         delta = (2 * self.L * y2) / (L_bar**2)

#         delta = np.clip(
#                     delta,
#                     self.delta_min,
#                     self.delta_max
#                 )


#         return np.array(
#             [
#                 self.v,
#                 delta
#             ]
#         )

import numpy as np


class PurePursuitController:
    """Projected Lookahead Pure Pursuit Controller."""

    def __init__(
        self,
        wheelbase: float,
        k_lookahead: float = 0.5,
        min_lookahead: float = 2.0,
        kp_speed: float = 1.0,
    ):
        self.L = wheelbase
        self.k = k_lookahead
        self.min_Ld = min_lookahead
        self.kp_v = kp_speed

        self.delta_min = np.deg2rad(-30.0)
        self.delta_max = np.deg2rad(30.0)

    def compute_control(
        self, state: np.ndarray, reference: np.ndarray
    ) -> np.ndarray:
        """Compute control command vector [acceleration, steering_angle].

        Parameters
        ----------
        state : ndarray
            Current state vector [x, y, theta, v].
        reference : ndarray
            Reference trajectory matrix of shape (N+1, 4).

        Returns
        -------
        control : ndarray
            Control command vector [acceleration, delta].
        """
        x, y, theta, v = state

        # Adaptive lookahead distance
        Ld = max(self.min_Ld, self.k * v)

        # 1. Project virtual point P along current vehicle heading
        Px = x + Ld * np.cos(theta)
        Py = y + Ld * np.sin(theta)

        # 2. Find nearest reference point C from projected point P
        dists_to_P = np.hypot(reference[:, 0] - Px, reference[:, 1] - Py)
        idx = np.argmin(dists_to_P)

        Cx, Cy = reference[idx, 0], reference[idx, 1]
        target_v = reference[idx, 3]
        L_bar = max(dists_to_P[idx], 1e-3)

        # 3. Calculate perpendicular error (y2) from C to vehicle heading line VP
        a = y - Py
        b = Px - x
        c = x * Py - Px * y
        y2 = (a * Cx + b * Cy + c) / np.hypot(a, b)

        # 4. Steering angle computation
        delta = (2.0 * self.L * y2) / (L_bar**2)
        delta = np.clip(delta, self.delta_min, self.delta_max)

        # 5. Longitudinal velocity tracking P-control
        acceleration = self.kp_v * (target_v - v)

        return np.array([acceleration, delta], dtype=np.float64)