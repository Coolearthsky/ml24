# Quantization-aware training

This compares two MNIST scratch-trained models, one using float32, and one using int8.

The architecture is one convolutional layer, pooling, and one dense layer for one-hot classifier output:

```
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 quantize_layer              (None, 28, 28)            3         
 quant_reshape               (None, 28, 28, 1)         1         
 quant_conv2d                (None, 26, 26, 12)        147       
 quant_max_pooling2d         (None, 13, 13, 12)        1         
 quant_flatten               (None, 2028)              1         
 quant_dense                 (None, 10)                20295     
=================================================================
```


Both models perform with about 96% accuracy.

The int8 is then exported to tflite, so it's about 1/4 the size.

The code is cribbed from
[the example colab](https://www.tensorflow.org/model_optimization/guide/quantization/training_example).

