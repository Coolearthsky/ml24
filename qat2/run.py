#! /usr/bin/python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from six import BytesIO
from IPython import display
from urllib.request import urlopen

import numpy as np
import tensorflow as tf

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
tf.get_logger().setLevel(absl.logging.ERROR)
