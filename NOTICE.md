# Third-party material

## PythonRobotics (MIT)

The Frenet-frame parts of `planning/local_path_planning/` are based on
[PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics):

| File | Origin |
|---|---|
| `cartesian_frenet_converter.py` | PythonRobotics `CartesianFrenetConverter` (Wang Zheng), **unchanged** |
| `frenet_trajectory_generator.py` | candidate generation after PythonRobotics' Frenet optimal trajectory (Atsushi Sakai), extended (modes, batch generation) |
| `frenet_vectorized.py` | the converter's Frenet-to-Cartesian formulas, vectorized |

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

## Reference

M. Werling, J. Ziegler, S. Kammel, S. Thrun, "Optimal Trajectory Generation for
Dynamic Street Scenarios in a Frenet Frame", ICRA 2010.

## Data

The [inD dataset](https://www.ind-dataset.com) (ika, RWTH Aachen University) is
not part of this repository. It is free for non-commercial use after
registration; its terms apply to the data, the background images and models
trained on them. `map/drivable_mask_images/18_mask.png` is a hand-drawn binary
mask for recording 18 and contains no dataset pixels.
