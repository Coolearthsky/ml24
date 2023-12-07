#! /bin/sh

pythin3 -m pip install --upgrade pip
python3 -m pip install mediapipe
python3 -m pip install mediapipe-model-maker

wget -q -O image.jpg https://storage.googleapis.com/mediapipe-tasks/object_detector/cat_and_dog.jpg

wget https://storage.googleapis.com/mediapipe-tasks/object_detector/android_figurine.zip
unzip android_figurine.zip
