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


# note the data is massaged a bit, adding height and width, so i checked it in.
#echo
#echo FETCHING DATA
#echo
#
#wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg
#wget https://storage.googleapis.com/mediapipe-tasks/object_detector/android_figurine.zip
#unzip android_figurine.zip

echo
echo CREATE INPUT RECORDS
echo
# note i think this is not required, the json etc can be read directly??

python3 -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir='./android_figurine/train/images' \
  --object_annotations_file='./android_figurine/train/labels.json' \
  --output_file_prefix='./tfrecords/train' \
  --num_shards=1

python3 -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir='./android_figurine/validation/images' \
  --object_annotations_file='./android_figurine/validation/labels.json' \
  --output_file_prefix='./tfrecords/validation' \
  --num_shards=1
