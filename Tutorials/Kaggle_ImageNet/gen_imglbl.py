''''
# Support
GitHub issues will be used for tracking requests and bugs. For questions go to [forums.xilinx.com](http://forums.xilinx.com/).
# License
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0]( http://www.apache.org/licenses/LICENSE-2.0 )
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
Copyright 2022 Xilinx, Inc.</sup></p>
'''

import pathlib

# put the classes into a list
classes = []
with open('LOC_synset_mapping.txt', 'r') as f:
    for line in f:
        line = line.replace('\n', '').split(' ')
        classes.append(line[0])

# create the image-label association files
print('Creating \'trn_imglbl.txt\' (image-label association file for training)...')
with open('trn_imglbl.txt', 'w') as ftrn:
    with open('./ILSVRC/ImageSets/CLS-LOC/train_cls.txt', 'r') as f:
        for line in f:
            line = line.replace('\n', '').split(' ')
            img_path = line[0].split('/')
            wnid = img_path[0]
            img = img_path[1] + '.JPEG'
            idx = classes.index(wnid)
            ftrn.write(f"{wnid + '/' + img} {idx}\n")

print('Creating \'val_imglbl.txt\' (image-label association file for validation)...')
val_dir = pathlib.Path('ILSVRC/Data/CLS-LOC/val')
vimg_list = list(val_dir.glob('*/*.JPEG'))
with open('val_imglbl.txt', 'w') as fval:
    for v in vimg_list:
        vpath = pathlib.PurePath(v)
        vpath_len = len(vpath.parts)
        wnid = vpath.parts[vpath_len - 2]
        img = vpath.parts[vpath_len - 1]
        idx = classes.index(wnid)
        fval.write(f"{wnid + '/' + img} {idx}\n")
