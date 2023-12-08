# Transfer Learning

The [TensorFlow tutorial about transfer learning](https://www.tensorflow.org/tutorials/images/transfer_learning)
covers image classification using MobileNetV2 using fine-tuning with a few thousand labeled images.

The general idea is to ablate the top layer and replace it with one specific to the new task.  Then freeze
all the layers except the new top one, and train a bit.  Then unfreeze some more layers for a bit more accuracy.

I ran 10 epochs with one free layer followed by 10 more epochs with 50 free layers, as suggested by the
tutorial, and it took about 5 minutes, resulting in about 98% accuracy.
