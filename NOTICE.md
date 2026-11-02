# Third-party material

## PythonRobotics (MIT)

The Frenet-frame parts of `planning/local_path_planning/` are based on
[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics):

| File | Origin |
|---|---|
| `cartesian_frenet_converter.py` | PythonRobotics `CartesianFrenetConverter` (Wang Zheng), unchanged |
| `frenet_trajectory_generator.py` | PythonRobotics Frenet optimal trajectory candidate generation (Atsushi Sakai), extended with additional modes and batched generation |
| `frenet_vectorized.py` | Vectorized version of the Frenet-to-Cartesian formulas |

The applicable MIT license notice is reproduced below:

```
MIT License

Copyright (c) 2016 - 2022 Atsushi Sakai and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Research reference

M. Werling, J. Ziegler, S. Kammel, and S. Thrun, “Optimal Trajectory Generation
for Dynamic Street Scenarios in a Frenet Frame,” IEEE International Conference
on Robotics and Automation, 2010.

## inD dataset

The [inD dataset](https://levelxdata.com/ind-dataset/) by ika, RWTH Aachen
University, is not included in this repository. Access and use are governed by
the dataset's separate terms. The official terms allow non-commercial research
use after access approval, require attribution, and prohibit distribution of
the dataset or modified versions. They permit qualifying abstract derivatives
only when they do not contain inD data or allow its recovery; distribution and
use remain non-commercial.

A model trained on inD may qualify as a derivative, but its distribution still
has to meet those conditions. Background images and recorded tracks are inD
data. The official terms permit only small portions of data for illustration
in qualifying research publications. Check the terms before publishing a GIF or
other output containing inD imagery or track data, and cite the dataset paper:

J. Bock et al., “The inD Dataset: A Drone Dataset of Naturalistic Road User
Trajectories at German Intersections,” IEEE Intelligent Vehicles Symposium,
2020, pp. 1929–1934, [doi:10.1109/IV47402.2020.9304839](https://doi.org/10.1109/IV47402.2020.9304839).

`map/drivable_mask_images/18_mask.png` is a hand-drawn binary drivable-area
annotation for recording 18 and contains no background-image pixels. The inD
terms still apply to dataset-derived material; refer to the official terms.
