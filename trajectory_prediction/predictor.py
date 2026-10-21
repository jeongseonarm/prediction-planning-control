"""
Trajectory prediction for the simulation.

    predictor = TrajectoryPredictor(data_path, log_dir)
    frames    = predictor.frames(start, end)
    tracks    = predictor.predict(frame)      # one dict per vehicle in `frame`

Each dict holds
    id          track id
    track_data  DataFrame, the last `input_frame` rows of the track up to `frame`
    gt_traj     (output_frame, 2) ground truth, or None if the future is incomplete
    pred_traj   (output_frame, 2) network prediction, or None if the history is short
    plan_traj   what the planner should use: pred_traj, or a constant velocity
                extrapolation of the history where there is no prediction
    size        (length, width) [m] or None
    heading     current heading [rad] or None
"""

import glob
import os

import numpy as np
import pandas as pd
import torch

from . import _HERE  # noqa: F401  (puts Models/ and Utils/ on sys.path)
from Models.Dual_transformer import Dual_transformer
from Utils.Base import Read_config, reverse_standard_scale
from Utils.Make_tensor import Make_tensor

_FEATURES = [
    "xCenter", "yCenter", "heading", "xVelocity", "yVelocity",
    "xAcceleration", "yAcceleration",
]
_VERIFY_TRACKS = 20  # tracks compared against Make_tensor before it is trusted


def constant_velocity_prediction(track_data, steps, frame_dt):
    """Fallback prediction for tracks the network cannot predict (short history)."""
    xy = track_data[["xCenter", "yCenter"]].to_numpy()

    if len(xy) == 0:
        return None

    if len(xy) >= 2:
        n = min(len(xy) - 1, 5)
        velocity = (xy[-1] - xy[-1 - n]) / (n * frame_dt)
    else:
        velocity = np.zeros(2)

    t = np.arange(1, steps + 1) * frame_dt
    return xy[-1] + velocity[None, :] * t[:, None]


def vehicle_heading(track_data):
    """Current heading of a track in radians (same convention as the ego yaw)."""
    try:
        return float(np.deg2rad(track_data["heading"].iloc[-1]))
    except (KeyError, IndexError):
        return None


def vehicle_size(track_data):
    """(length, width) of a track in metres, or None if the columns are missing."""
    try:
        return (
            float(track_data["length"].iloc[-1]),
            float(track_data["width"].iloc[-1]),
        )
    except (KeyError, IndexError):
        return None


class TrajectoryPredictor:
    def __init__(
        self,
        data_path,
        log_dir,
        input_frame=25,
        output_frame=75,
        frame_dt=0.04,
        device="cpu",
    ):
        self.input_frame = input_frame
        self.output_frame = output_frame
        self.frame_dt = frame_dt

        # ---- data ------------------------------------------------------
        data = pd.read_csv(data_path)
        data = data.sort_values(["frame", "trackId"]).reset_index(drop=True)
        self.data = data

        # every track as one block of rows (ascending frame), and the vehicles
        # present in each frame, so a frame never has to scan the whole table
        self._tracks = {}
        for track_id, rows in data.groupby("trackId", sort=False):
            self._tracks[track_id] = (rows, rows["frame"].to_numpy())

        self._ids_in_frame = {
            frame: ids.unique() for frame, ids in data.groupby("frame", sort=False)["trackId"]
        }

        # ---- model -----------------------------------------------------
        config_files = sorted(glob.glob(os.path.join(log_dir, "*.json")))
        self.config = {}
        for file in config_files:
            self.config.update(Read_config(file))
        self.config["Device"] = device

        self.device = torch.device(device)
        self.model = Dual_transformer(self.config).to(self.device)

        weight_dir = os.path.join(log_dir, "pt")
        weight_files = glob.glob(os.path.join(weight_dir, "*.pt"))

        if len(weight_files) == 0:
            raise FileNotFoundError(f"No model weight file found in: {weight_dir}")

        latest_weight = max(weight_files, key=os.path.getmtime)

        self.model.load_state_dict(
            torch.load(latest_weight, map_location=self.device)
        )
        self.model.eval()

        # numpy version of Make_tensor, enabled only if it reproduces it exactly
        self._mean = np.asarray(self.config["mean"][:len(_FEATURES)], dtype=np.float64)
        self._std = np.asarray(self.config["std"][:len(_FEATURES)], dtype=np.float64)
        self._fast_tensor = True
        self._verified = 0

    def frames(self, start, end):
        """Sorted frames with start <= frame < end."""
        return [
            frame for frame in sorted(self.data["frame"].unique())
            if start <= frame < end
        ]

    def _window(self, track_id, lower, lower_side, upper, upper_side):
        rows, frames = self._tracks[track_id]
        lo = np.searchsorted(frames, lower, side=lower_side)
        hi = np.searchsorted(frames, upper, side=upper_side)
        return rows.iloc[lo:hi]

    def _make_tensor(self, track_data):
        """Make_tensor(track_data, config), with a faster numpy path that is
        checked against it on the first tracks and dropped if it ever differs."""
        if not self._fast_tensor:
            return Make_tensor(track_data, self.config)

        values = track_data[_FEATURES].to_numpy(dtype=np.float64)
        fast = torch.tensor(
            (values - self._mean) / self._std,
            device=self.config["Device"], dtype=torch.float32,
        )

        if self._verified < _VERIFY_TRACKS:
            reference = Make_tensor(track_data, self.config)
            if not torch.equal(fast, reference):
                self._fast_tensor = False
                return reference
            self._verified += 1

        return fast

    def _network(self, track_data):
        tensor_data = self._make_tensor(track_data)

        with torch.no_grad():
            trajectory, *_ = self.model(
                tensor_data.unsqueeze(0),
                tensor_data.unsqueeze(0),
                tensor_data.unsqueeze(0),
            )

        trajectory = trajectory[:, :self.output_frame, :2]

        trajectory[:, :, 0] = reverse_standard_scale(
            trajectory[:, :, 0], self.config["mean"][0], self.config["std"][0]
        )
        trajectory[:, :, 1] = reverse_standard_scale(
            trajectory[:, :, 1], self.config["mean"][1], self.config["std"][1]
        )

        return trajectory.squeeze(0).detach().cpu().numpy()

    def predict(self, frame):
        result = []

        for track_id in self._ids_in_frame.get(frame, []):

            # history: frame - input_frame < f <= frame
            track_data = self._window(
                track_id, frame - self.input_frame, "right", frame, "right"
            ).copy()

            # future: frame + 1 <= f <= frame + output_frame
            gt_data = self._window(
                track_id, frame + 1, "left", frame + self.output_frame, "right"
            ).copy()

            gt_traj = gt_data[["xCenter", "yCenter"]].to_numpy()

            pred_traj = None
            if len(track_data) == self.input_frame:
                pred_traj = self._network(track_data)

            # the ground truth is only for drawing / evaluation
            if len(gt_data) != self.output_frame:
                gt_traj = None

            if pred_traj is not None:
                plan_traj = pred_traj
            else:
                plan_traj = constant_velocity_prediction(
                    track_data, self.output_frame, self.frame_dt
                )

            result.append(
                {
                    "id": track_id,
                    "track_data": track_data,
                    "gt_traj": None if gt_traj is None else gt_traj.copy(),
                    "pred_traj": None if pred_traj is None else pred_traj.copy(),
                    "plan_traj": plan_traj,
                    "size": vehicle_size(track_data),
                    "heading": vehicle_heading(track_data),
                }
            )

        return result
