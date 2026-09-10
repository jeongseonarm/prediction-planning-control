# Prediction-Planning-Control

<!-- ![Build](https://img.shields.io/badge/build-passing-brightgreen) ![License](https://img.shields.io/badge/license-TBD-lightgrey) -->

Python implementation of an autonomous driving pipeline on the **inD dataset**: trajectory prediction, global path planning, and vehicle control, integrated into a single simulation with animated visualization.

<!-- Teaser image / demo GIF -->
<!--
![Teaser](docs/teaser.gif)
-->

---

## What is this?

This repository combines three components of a self-driving stack into one runnable simulation:

- **Trajectory prediction** of surrounding vehicles (Dual Transformer)
- **Global path planning** for the ego vehicle (Hybrid A*)
- **Vehicle control** to track the planned path (MPC / Pure Pursuit)

> **Status:** trajectory prediction and control currently run **in parallel, independently** — predicted trajectories are not yet consumed by the control loop. Local planning is not implemented yet; see [Pipeline](#pipeline) and [Roadmap](#roadmap--limitations) below.

---

## Pipeline

```mermaid
flowchart LR
    A["Drivable Mask +\nCollision Checker"] --> B["Global Path Planning\n(Hybrid A*)"]
    B --> C["Speed Profile"]
    C -.->|"not yet implemented"| L["Local Planning"]
    L -.-> D
    C --> D["Vehicle Control\n(MPC / Pure Pursuit)"]
    E["inD Track Data"] --> F["Trajectory Prediction\n(Dual Transformer)"]
    F -.->|"not yet connected"| L
    D --> G["Visualization"]
    F --> G

    style L stroke-dasharray: 5 5
```

| Stage | Method | Status |
|---|---|---|
| Global path planning | Hybrid A* over a drivable-area mask | ✅ implemented |
| Speed profile | Target-speed profile along the planned path | ✅ implemented |
| Local planning | Reactive planning informed by predicted trajectories | 🔲 planned |
| Control | MPC (default), Pure Pursuit as an alternative | ✅ implemented |
| Trajectory prediction | Dual Transformer, evaluated against ground truth | ✅ implemented |
| Visualization | matplotlib animation | ✅ implemented |

---

## Getting Started

### Requirements

```bash
git clone https://github.com/jeongseonarm/prediction-planning-control.git
cd prediction-planning-control
pip install -r requirements.txt
```

Key dependencies: PyTorch 1.13.1 (CPU), NumPy, Pandas, OpenCV, matplotlib, tqdm.

### Data Preparation

Required data (see path constants at the top of `main.py`):

| Path | Contents |
|---|---|
| `trajectory_prediction/Data/` | inD track CSV + scene background image |
| `map/drivable_mask_images/` | drivable-area mask image for the scene |
| `trajectory_prediction/log/` | prediction model config (`.json`) and trained weights (`.pt`) |

Pretrained weights and the inD dataset are **not included** in this repo and must be obtained separately.

### Run

```bash
python main.py
```

### Key Parameters (`main.py`)

| Parameter | Description |
|---|---|
| `DATA_NUM` | inD scene/recording number to load |
| `EGO_START`, `EGO_GOAL` | ego vehicle start/goal state `[x, y, theta]` |
| `TARGET_SPEED` | target cruising speed |
| `MPC_HORIZON` | MPC prediction horizon |
| `INPUT_FRAME`, `OUTPUT_FRAME` | input/output frame length for trajectory prediction |
| `PREDICTION_START_FRAME`, `PREDICTION_END_FRAME` | frame range simulated |

---

## Project Structure

```
prediction-planning-control/
├── trajectory_prediction/
│   ├── Data/                    # inD track CSVs, background images
│   ├── log/                     # training configs (json) and model weights (.pt)
│   ├── Models/
│   │   └── Dual_transformer.py  # trajectory prediction model
│   └── Utils/
│       ├── Average_deviation.py
│       ├── Base.py              # config loading, scale un-normalization
│       ├── Draw_trajectory.py
│       ├── Draw_vehicle.py
│       └── Make_tensor.py
├── planning/
│   ├── global_path_planning/
│   │   ├── hybrid_a_star.py     # Hybrid A* global planner
│   │   └── collision_checker.py # mask-based collision checking
│   └── local_planning/          # planned — not implemented yet
├── control/
│   ├── controllers/
│   │   ├── mpc.py               # Model Predictive Control
│   │   └── pure_pursuit.py      # Pure Pursuit controller
│   └── models/
│       └── kinematic_bicycle.py # vehicle kinematics model
├── map/
│   └── drivable_mask_images/    # drivable-area mask images
├── main.py                      # pipeline entry point
├── requirements.txt
└── .gitignore
```

---

## Results

### Quantitative

<!--
| Metric | Value |
|---|---|
| Average Deviation (ADE) | TBD |
-->

### Qualitative

<!--
![Demo](docs/demo.gif)
-->

---

## Roadmap / Limitations

- [ ] **Local planner (`planning/local_planning/`).** Not implemented yet. Will take predicted trajectories of surrounding vehicles as input to perform reactive planning (e.g. collision avoidance, replanning) between the global path and the controller.
- [ ] Integrate trajectory prediction output into the control loop, instead of running prediction and control in parallel.

## License

TBD — no license has been chosen yet. Without one, default copyright rules apply (others cannot legally reuse/redistribute the code). Add a `LICENSE` file (e.g. MIT, Apache-2.0) once you decide.
