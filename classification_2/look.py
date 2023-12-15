#! /usr/bin/python3


import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np
from tensorflow import keras

import matplotlib
import matplotlib.pyplot as plt


from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib
from official.vision.ops.preprocess_ops import normalize_image
from official.vision.ops.preprocess_ops import resize_and_crop_image
from official.vision.utils.object_detection import visualization_utils
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder


matplotlib.use('TkAgg')

category_index={
    1: {
        'id': 1,
        'name': 'blue'
       },
    2: {
        'id': 2,
        'name': 'red'
       }
}
tf_ex_decoder = TfExampleDecoder()


def show_batch(raw_records, num_of_examples):
  plt.figure(figsize=(20, 20))
  use_normalized_coordinates=True
  min_score_thresh = 0.30
  for i, serialized_example in enumerate(raw_records):
    plt.subplot(1, 3, i + 1)
    decoded_tensors = tf_ex_decoder.decode(serialized_example)
    image = decoded_tensors['image'].numpy().astype('uint8')
    scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image,
        decoded_tensors['groundtruth_boxes'].numpy(),
        decoded_tensors['groundtruth_classes'].numpy().astype('int'),
        scores,
        category_index=category_index,
        use_normalized_coordinates=use_normalized_coordinates,
        max_boxes_to_draw=200,
        min_score_thresh=min_score_thresh,
        agnostic_mode=False,
        instance_masks=None,
        line_thickness=4)

    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Image-{i+1}')
  plt.show()



print("reading")
raw_records = tf.data.TFRecordDataset('./test/balls.tfrecord').shuffle(buffer_size=20).take(3)

print("showing")
show_batch(raw_records, 3)

#for raw_record in raw_records:
#  example = tf.train.Example()
#  example.ParseFromString(raw_record.numpy())
#  print(example)

