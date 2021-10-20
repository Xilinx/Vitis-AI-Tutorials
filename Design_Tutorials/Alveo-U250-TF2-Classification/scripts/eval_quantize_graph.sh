# Copyright 2021 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

python train_eval_h5.py --model tf2_resnet50/vai_q_output/quantized.h5 \
					      --quantize_eval=true \
					      --eval_only=true \
					      --eval_images=false \
					      --eval_image_path=/data3/datasets/Kaggle/fruits-360/val_for_tf2 \
					      --eval_image_list=/data3/datasets/Kaggle/fruits-360/val_labels.txt \
					      --label_offset=1 \
					      --gpus 0 \
					      --eval_batch_size=50
