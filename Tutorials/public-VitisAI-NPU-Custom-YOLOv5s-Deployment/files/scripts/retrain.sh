# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Daniele.Bagni@amd.com
# date: 20 Feb 2025



SIZE=640
BATCH_SIZE=32 # 1  #32 
EPOCHS=20     #30  #20 
MODEL="/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files/yolov5/weights/yolov5s.pt"
WORKERS=1
PROJECT="weights"
RUN_NAME=${MODEL}_size${SIZE}_epochs${EPOCHS}_batch${BATCH_SIZE}_small
DATA_YAML=/workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files/yolov5/data.yaml


python3 train.py --img ${SIZE} --batch ${BATCH_SIZE} --epochs ${EPOCHS} \
    --data ${DATA_YAML} --weights ${MODEL} --workers ${WORKERS}  --name ${RUN_NAME} --exist-ok --project ${PROJECT}



#python3 train.py --img 640 --batch 1 --epochs 1 --data /workspace/tutorials/VitisAI-NPU-Custom-YOLOv5s-Deployment/files/yolov5/data.yaml --weights ../home_demo_yolov5/weights/yolov5s.pt --workers 1 --name ../home_demo_yolov5/weights/yolov5s.pt_size640_epochs1_batch1_small --exist-ok
