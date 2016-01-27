from __future__ import division
import numpy as np

COLORS = {'brain': np.array([170, 68, 153]) / 255,   # violet
          'er': np.array([204, 102, 119]) / 255,     # red
          'random': np.array([153, 153, 51]) / 255,  # gold; same as config
          'configuration': np.array([153, 153, 51]) / 255,  # gold
          'rand': np.array([153, 153, 51]) / 255,           # gold
          'small-world': np.array([17, 119, 51]) / 255,     # green
          'scale-free': np.array([51, 34, 136]) / 255,      # dark blue
          'sgpa': np.array([136, 204, 238]) / 255,          # cyan
          'sg': np.array([136, 204, 238]) / 255,            # cyan
          'ta': np.array([225, 25, 25]) / 255,              # pinkish red
          'tapa': np.array([225, 25, 25]) / 255,            # pinkish red
          'pa': np.array([0, 0, 0]) / 255,                  # black
          'geom': np.array([0, 0, 0]) / 255}                # black

FACE_COLOR = 'w'
AX_COLOR = 'k'
FONT_SIZE = 20
