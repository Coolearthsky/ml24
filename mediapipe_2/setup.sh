#! /bin/sh

python3 -m pip install --upgrade pip
python3 -m pip install mediapipe-model-maker

wget https://storage.googleapis.com/mediapipe-tasks/object_detector/android_figurine.zip
unzip android_figurine.zip
