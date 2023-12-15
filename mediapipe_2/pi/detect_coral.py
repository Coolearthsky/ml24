# Original at https://github.com/googlesamples/mediapipe/blob/main/examples/object_detection/raspberry_pi/detect.py
#
# this contains some modifications:
#
# different model name, model_int_qat.tflite which is the QAT trained model from https://developers.google.com/mediapipe/solutions/customization/object_detector
#
# different main loop,
#
"""Main scripts to run object detection."""

import argparse
import sys
import time

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize

from enum import Enum

import libcamera
import msgpack
import numpy as np
import ntcore
import os

from cscore import CameraServer

# from ntcore import NetworkTableInstance
from picamera2 import Picamera2
from pupil_apriltags import Detector


# from https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter

import copy


# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()


def run(
    model: str,
    max_results: int,
    score_threshold: float,
    camera_id: int,
    width: int,
    height: int,
) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
      model: Name of the TFLite object detection model.
      max_results: Max number of detection results.
      score_threshold: The score threshold of detection results.
      camera_id: The camera id to be passed to OpenCV.
      width: The width of the frame captured from the camera.
      height: The height of the frame captured from the camera.
    """
    print("RUN")

    fullwidth = 1664
    fullheight = 1232
    width = 832
    height = 616

    camera = Picamera2()

    camera_config = camera.create_still_configuration(
        # one buffer to write, one to read, one in between so we don't have to wait
        buffer_count=6,
        main={
            "format": "YUV420",
            "size": (fullwidth, fullheight),
        },
        lores={"format": "YUV420", "size": (width, height)},
        controls={
            "FrameDurationLimits": (5000, 33333),  # 41 fps
            "NoiseReductionMode": libcamera.controls.draft.NoiseReductionModeEnum.Off,
        },
    )

    print("REQUESTED")
    print(camera_config)
    camera.align_configuration(camera_config)
    print("ALIGNED")
    print(camera_config)
    camera.configure(camera_config)
    print(camera.camera_controls)


    camera.start()

    # Start capturing video input from the camera
    #    cap = cv2.VideoCapture(camera_id)
    #    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #
    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_frame = None
    detection_result_list = []

    def save_result(
        result: vision.ObjectDetectorResult,
        unused_output_image: mp.Image,
        timestamp_ms: int,
    ):
        global FPS, COUNTER, START_TIME

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        detection_result_list.append(result)
        COUNTER += 1

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result,
    )
    detector = vision.ObjectDetector.create_from_options(options)

# see https://github.com/google-coral/pycoral/blob/master/examples/detect_image.py

    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()

    if common.input_details(interpreter, 'dtype') != np.uint8:
      raise ValueError('Only support uint8 input type.')
    size = common.input_size(interpreter)
    print("input size ", size)
    params = common.input_details(interpreter, 'quantization_parameters')
    scale = params['scales']
    zero_point = params['zero_points']
    mean = 128 
    std = 128


    # Continuously capture images from the camera and run inference
    #    while cap.isOpened():
    #        success, image = cap.read()
    #        if not success:
    #            sys.exit(
    #                "ERROR: Unable to read from webcam. Please verify your webcam settings."
    #            )

    output_stream = CameraServer.putVideo("Processed", width, height)

    try:
        while True:
            request = camera.capture_request()
            try:





                #print("request0")
                # logic goes here
                #buffer = request.make_buffer("lores")
                buffer = request.make_array("lores")

                img = buffer
                image = cv2.cvtColor(img, cv2.COLOR_YUV420p2BGR)

                # img = img[:height, :width]

                # image = img
                # image = cv2.flip(image, 1)

                #print("request1")
                #print(img)

                # Convert the image from BGR to RGB as required by the TFLite model.
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                coral_image = cv2.resize(rgb_image, size)

                #cheight, cwidth, _ = coral_image.shape
                #_, scale = common.set_resized_input(interpreter, (cwidth, cheight), lambda size: cv2.resize(coral_image, size))
                common.set_input(interpreter, coral_image)

                interpreter.invoke()

# https://github.com/tensorflow/tensorflow/issues/51591
                #objs = detect.get_objects(interpreter, 0.5)
                #print("objs ",objs.shape)
                boxes = copy.copy(common.output_tensor(interpreter, 1))[0]
                #classes = common.output_tensor(interpreter, 3)
                scores = copy.copy(common.output_tensor(interpreter, 0))[0]
                #count = common.output_tensor(interpreter, 2)
                print("boxes ", boxes.shape)
                print("box0 ", boxes[0])
                print("scores ",  scores.shape)
                print("score0 ", scores[0])





                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)

                #print("request1.5")

                # Run object detection using the model.
                detector.detect_async(mp_image, time.time_ns() // 1_000_000)

                #print("request2")

                # Show the FPS
                fps_text = "FPS = {:.1f}".format(FPS)
                text_location = (left_margin, row_size)
                #current_frame = image
                current_frame = rgb_image
                cv2.putText(
                    current_frame,
                    fps_text,
                    text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

                #print("request3")
                if detection_result_list:
                    # print(detection_result_list)
                    current_frame = visualize(current_frame, detection_result_list[0])
                    detection_frame = current_frame
                    detection_result_list.clear()

                #print("request4")
                if detection_frame is not None:
                    # cv2.imshow("object_detection", detection_frame)
                    output_stream.putFrame(detection_frame)
                #print("request5")

                # Stop the program if the ESC key is pressed.
            #                if cv2.waitKey(1) == 27:
            #                    break

            finally:
                request.release()
                #print("release")
    finally:
        camera.stop()
        detector.close()
        print("all done")
    # cap.release()
    # cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        help="Path of the object detection model.",
        required=False,
        default="model_int8_qat.tflite",
    )
    parser.add_argument(
        "--maxResults",
        help="Max number of detection results.",
        required=False,
        default=5,
    )
    parser.add_argument(
        "--scoreThreshold",
        help="The score threshold of detection results.",
        required=False,
        type=float,
        default=0.25,
    )
    # Finding the camera ID can be very reliant on platform-dependent methods.
    # One common approach is to use the fact that camera IDs are usually indexed sequentially by the OS, starting from 0.
    # Here, we use OpenCV and create a VideoCapture object for each potential ID with 'cap = cv2.VideoCapture(i)'.
    # If 'cap' is None or not 'cap.isOpened()', it indicates the camera ID is not available.
    parser.add_argument(
        "--cameraId", help="Id of camera.", required=False, type=int, default=0
    )
    parser.add_argument(
        "--frameWidth",
        help="Width of frame to capture from camera.",
        required=False,
        type=int,
        default=1280,
    )
    parser.add_argument(
        "--frameHeight",
        help="Height of frame to capture from camera.",
        required=False,
        type=int,
        default=720,
    )
    args = parser.parse_args()

    run(
        args.model,
        int(args.maxResults),
        args.scoreThreshold,
        int(args.cameraId),
        args.frameWidth,
        args.frameHeight,
    )


if __name__ == "__main__":
    main()
