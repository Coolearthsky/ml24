#! /usr/bin/python3

#from google.colab import files
import os
import json
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector

train_dataset_path = "android_figurine/train"
validation_dataset_path = "android_figurine/validation"

with open(os.path.join(train_dataset_path, "labels.json"), "r") as f:
  labels_json = json.load(f)
for category_item in labels_json["categories"]:
  print(f"{category_item['id']}: {category_item['name']}")

