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

python train_eval_h5.py --model float/resnet_50.h5 \
					      --eval_only=false \
						  --createnewmodel=true \
					      --eval_images=false \
					      --eval_image_path=/data3/datasets/Kaggle/fruits-360/tf_records \
					      --label_offset=1 \
					      --gpus 0,1 \
					      --eval_batch_size=100
