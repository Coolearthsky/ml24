#! /usr/bin/python3

# this is https://github.com/tensorflow/models/blob/master/official/projects/qat/vision/train.py

# import tensorflow as tf
# from tensorflow import keras

# fashion_mnist = tf.keras.datasets.fashion_mnist

from absl import app

from official.common import flags as tfm_flags
from official.projects.qat.vision import registry_imports
from official.vision import train


if __name__ == "__main__":
    tfm_flags.define_flags()
    app.run(train.main)
