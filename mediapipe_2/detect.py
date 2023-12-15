#! /usr/bin/python3

import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

#MARGIN = 10  # pixels
#ROW_SIZE = 10  # pixels
#FONT_SIZE = 10
#FONT_THICKNESS = 10
#TEXT_COLOR = (255, 255, 255)


def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """

  for detection in detection_result.detections:
    # Draw bounding_box
    bbox = detection.bounding_box
    start_point = bbox.origin_x, bbox.origin_y
    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    #cv2.rectangle(image, start_point, end_point, (255,255,255), 2)
    cv2.rectangle(image, start_point, end_point, (255,255,255), 8)

    # Draw label and score
    category = detection.categories[0]
    category_name = category.category_name
    probability = round(category.score, 2)
    result_text = category_name + ' (' + str(probability) + ')'
    text_location = (bbox.origin_x + 10, bbox.origin_y + 20)
    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 10, (255,255,255), 10)
    #cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

  return image


# try the float32 model first.

#model_path = '/home/joel/FRC/TRUHER/ml24/mediapipe_2/exported_model/model.tflite'
model_path = '/home/joel/FRC/TRUHER/ml24/mediapipe_2/exported_model/model_int8_qat.tflite'

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    score_threshold=0.5,
    max_results=5,
    running_mode=VisionRunningMode.IMAGE)

with ObjectDetector.create_from_options(options) as detector:
  # The detector is initialized. Use it here.
  # ...
  for file in list(glob.glob('/home/joel/FRC/TRUHER/ml24/mediapipe_2/android_figurine/validation/images/*.jpg')):
  #for file in list(glob.glob('/home/joel/FRC/TRUHER/ml24/mediapipe_2/android_figurine/train/images/*.jpg')):
  
  #mp_image = mp.Image.create_from_file('/home/joel/FRC/TRUHER/ml24/mediapipe_2/android_figurine/validation/images/IMG_0498.jpg')
    mp_image = mp.Image.create_from_file(file)
    detection_result = detector.detect(mp_image)
    
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    #rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    plt.imshow(annotated_image)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    # this doesn't work becuse somehow TF messes up cv2
    #cv2.imshow('annotated image', rgb_annotated_image)
