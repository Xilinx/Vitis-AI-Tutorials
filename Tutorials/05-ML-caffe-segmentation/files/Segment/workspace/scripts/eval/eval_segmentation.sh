if [[ -z "${CITYSCAPES_DATASET}" ]]; then
  export CITYSCAPES_DATASET=/workspace/Segment/Cityscapes
fi
python evaluate_cityscapes.py seg $CITYSCAPES_DATASET/test_gtFine_cls19/val results/ 2>&1 | tee logs/test_cityscape_`date +%y%m%d-%H%M%S`.log
