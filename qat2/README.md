# Quantization-Aware Training (QAT)

There are two ways to make a "quantized," i.e. integer, model for use in the Coral EdgeTPU hardware using tflite.

The first way is to build a "normal" floating point model, and then quantize it at the end.

A better way is to constrain the model during training, so that the weights are only ever allowed to take integral values.

Google added some "official" QAT models to the TensorFlow Model Garden
[for image classification in mid-2022](https://blog.tensorflow.org/2022/06/Adding-Quantization-aware-Training-and-Pruning-to-the-TensorFlow-Model-Garden.html)
and added [object detection models in December 2022](https://blog.tensorflow.org/2022/12/new-state-of-art-quantized-models-added-in-tf-model-garden.html),
using MobileNetV2 and RetinaNet, a widely-used object detection architecture.

The resulting architecture is a pyramid of bottlenecks (the MobileNet side) feeding classifiers and box regression (the RetinaNet side).

The code here follows the
[QAT tutorial](https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/qat/vision/docs/qat_tutorial.ipynb)
which is part of the [QAT project](https://github.com/tensorflow/models/tree/master/official/projects/qat/vision).

The training part is referenced [here](https://github.com/tensorflow/models/tree/master/official/projects/qat/vision#training),
but it references Cloud TPU, which I don't want, so there are some changes in the yaml config file.


To run it:

* sh setup.sh to install tensorflow, download the model, etc.
* sh train.sh to train the model
