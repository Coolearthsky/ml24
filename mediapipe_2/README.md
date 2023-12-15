# Mediapipe 2

One more try to use the media pipe thing for object detection.

https://developers.google.com/mediapipe/solutions/customization/object_detector

it uses the android figurine dataset, which is COCO json.  It can also read Pascal VOC,
which might come in handy for training our own data.

To run it:

* setup.py: download some data
* fetch.py: look at the labels
* viz.py: vizualize it
* train.py: retrain
* quant.py: quantization-aware training

Then I followed this page to do the inference:

https://developers.google.com/mediapipe/solutions/vision/object_detector/python

also the sample code:

https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/object_detection/python/object_detector.ipynb

Run it:

* detect.py: run detection 

using the float32 model, it's overfit, looks "great," total loss about 0.15.

the int8 model is terrible, total loss around 1; using the parameters from the guide improves things.
I also ran it for longer, because it seemed to 

* learning rate 0.1
* decay rate 1 (no decay)



Finally try it on the pi:

https://github.com/googlesamples/mediapipe/tree/main/examples/object_detection/raspberry_pi
