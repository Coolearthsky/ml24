#! /bin/sh

# see https://colab.research.google.com/github/tensorflow/models/blob/master/official/projects/qat/vision/docs/qat_tutorial.ipynb

echo
echo INSTALLING TENSORFLOW
echo
python3 -m pip install tensorflow

echo
echo INSTALLING TF-MODELS
echo
python3 -m pip install tf-models-official

echo
echo FETCHING MODEL
echo
curl https://storage.googleapis.com/tf_model_garden/vision/qat/mobilenetv2_ssd_coco/mobilenetv2_ssd_i256_ckpt.tar.gz --output model.tar.gz
tar -xvzf model.tar.gz
