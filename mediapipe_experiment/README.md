# TF Lite Experiment

This is the notes from Joel's attempt to get the tflite training pipeline
running on his Linux machine.

I think the situation is that the person inside Google who made tflite-model-maker has abandoned it,
so it has become broken as a result of Dependency Hell.

The only thing getting any attention is the "mediapipe" project.

* [Object detection overview](https://developers.google.com/mediapipe/solutions/vision/object_detector)
* [Object detection guide](https://developers.google.com/mediapipe/solutions/vision/object_detector/python)
* [Training](https://developers.google.com/mediapipe/solutions/customization/object_detector) This is what they call "customization" -- note that training is not supported for EfficientDet, so you have to use MobileNet.
* [Raspberry Pi example](https://github.com/googlesamples/mediapipe/blob/main/examples/object_detection/raspberry_pi/detect.py)
* [MediaPipe/Raspberry Pi DevRel page](https://developers.googleblog.com/2023/08/mediapipe-for-raspberry-pi-and-ios.html)
* [example detection colab](https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/object_detection/python/object_detector.ipynb) that actually works
* [recent work on Raspberry Pi support](https://github.com/google/mediapipe/issues/4744), version 0.10.8 seems to work on CPU

To play with the code here, run these in order:

* mediapipe-setup.sh
* mediapipe-viz.py (to see the training examples, note the filenames inside)
* mediapipe-train.py
* mediapipe-run.py (note the filenames inside, you can use this for various inputs and models)

More useful links:

* [Mediapipe ignores negative images](https://github.com/google/mediapipe/issues/4423), which makes fine-tuning basically impossible .. it's hard to believe this is actually true.
* [Tensorflow will never fix tflite-model-maker](https://github.com/tensorflow/tensorflow/issues/60431).
