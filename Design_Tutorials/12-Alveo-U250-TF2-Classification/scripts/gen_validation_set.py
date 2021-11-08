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

import numpy as np
import cv2
import os
labels= ["Apple Braeburn","Apple Crimson Snow","Apple Golden 1","Apple Golden 2","Apple Golden 3", "Apple Granny Smith","Apple Pink Lady","Apple Red 1", "Apple Red 2", 
"Apple Red 3","Apple Red Delicious","Apple Red Yellow 1", "Apple Red Yellow 2", "Apricot", "Avocado", "Avocado ripe","Banana","Banana Lady Finger","Banana Red","Beetroot", "Blueberry",
"Cactus fruit", "Cantaloupe 1", "Cantaloupe 2", "Carambula", "Cauliflower", "Cherry 1", "Cherry 2", "Cherry Rainier","Cherry Wax Black","Cherry Wax Red","Cherry Wax Yellow","Chestnut", 
"Clementine", "Cocos", "Corn", "Corn Husk","Cucumber Ripe", "Cucumber Ripe 2", "Dates", "Eggplant", "Fig", "Ginger Root","Granadilla", "Grape Blue", "Grape Pink", "Grape White", "Grape White 2", 
"Grape White 3", "Grape White 4", "Grapefruit Pink", "Grapefruit White","Guava", "Hazelnut", "Huckleberry", "Kaki", "Kiwi", "Kohlrabi", "Kumquats", "Lemon", "Lemon Meyer","Limes", "Lychee",
"Mandarine", "Mango", "Mango Red", "Mangostan", "Maracuja", "Melon Piel de Sapo","Mulberry","Nectarine", "Nectarine Flat","Nut Forest", "Nut Pecan", "Onion Red", "Onion Red Peeled","Onion White", 
"Orange", "Papaya", "Passion Fruit","Peach","Peach 2", "Peach Flat", "Pear","Pear 2", "Pear Abate","Pear Forelle", "Pear Kaiser", "Pear Monster", "Pear Red", "Pear Stone", "Pear Williams", 
"Pepino","Pepper Green","Pepper Orange","Pepper Red","Pepper Yellow", "Physalis", "Physalis with Husk","Pineapple", "Pineapple Mini","Pitahaya Red", "Plum", "Plum 2", "Plum 3", "Pomegranate","Pomelo Sweetie",
"Potato Red","Potato Red Washed","Potato Sweet", "Potato White", "Quince", "Rambutan", "Raspberry","Redcurrant","Salak", "Strawberry","Strawberry Wedge","Tamarillo","Tangelo","Tomato 1", "Tomato 2", 
"Tomato 3", "Tomato 4","Tomato Cherry Red","Tomato Heart","Tomato Maroon","Tomato Yellow", "Tomato not Ripened","Walnut","Watermelon"]

validation_images="/data2/datasets/Kaggle/fruits-360/Images/validation"
validation_output="/data2/datasets/Kaggle/fruits-360/val_for_tf2"
validation_labels="/data2/datasets/Kaggle/fruits-360/val_labels.txt"


validation_labels_file= open(validation_labels,"w")

index=1

for (dirpath, dirnames, filenames) in os.walk(validation_images):
    for filename in filenames:
        folder=dirpath.split(os.sep)
        folder = folder[-1]
        output_label_name = folder
        output_label_name = output_label_name.replace(" ","")
        label_idx = labels.index(folder)
        image = cv2.imread(dirpath+ '/'+ filename)
        output_filepath = validation_output + '/' + output_label_name + filename
        cv2.imwrite(output_filepath,image)
        validation_labels_file.write(output_label_name + filename + " " + str(label_idx) + "\n")
        print("wrote: ", output_filepath)
validation_labels_file.close()