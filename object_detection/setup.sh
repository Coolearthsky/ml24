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
echo FETCHING DATA
echo

curl -L 'https://public.roboflow.com/ds/ZpYLqHeT0W?key=ZXfZLRnhsc' > './BCCD.v1-bccd.coco.zip'
unzip -q -o './BCCD.v1-bccd.coco.zip' -d './BCC.v1-bccd.coco/'
rm './BCCD.v1-bccd.coco.zip'

echo
echo SPLITTING DATA
echo

python3 -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir='./BCC.v1-bccd.coco/train' \
  --object_annotations_file='./BCC.v1-bccd.coco/train/_annotations.coco.json' \
  --output_file_prefix='./bccd_coco_tfrecords/train' \
  --num_shards=1

python3 -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir='./BCC.v1-bccd.coco/valid' \
  --object_annotations_file='./BCC.v1-bccd.coco/valid/_annotations.coco.json' \
  --output_file_prefix='./bccd_coco_tfrecords/valid' \
  --num_shards=1

python3 -m official.vision.data.create_coco_tf_record --logtostderr \
  --image_dir='./BCC.v1-bccd.coco/test' \
  --object_annotations_file='./BCC.v1-bccd.coco/test/_annotations.coco.json' \
  --output_file_prefix='./bccd_coco_tfrecords/test' \
  --num_shards=1
