"""
Trajectory prediction (Dual Transformer on inD).

The network code keeps its own top-level package names (Models, Utils), so this
folder is put on sys.path once, here, instead of in every script.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
