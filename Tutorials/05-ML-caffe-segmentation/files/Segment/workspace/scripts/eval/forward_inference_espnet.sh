#!/usr/bin/env bash
if [[ -z "${CITYSCAPES_DATASET}" ]]; then
  export CITYSCAPES_DATASET=/workspace/Segment/Cityscapes
fi
rm -rf *.caffemodel *.prototxt
ln -s ../../model/espnet/deploy.prototxt deploy.prototxt
ln -s ../../model/espnet/final_models/pretrained.caffemodel deploy.caffemodel
python segmentation_miou_test.py

