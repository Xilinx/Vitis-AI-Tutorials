#!/bin/bash

#Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
#SPDX-License-Identifier: MIT

# date 28 Apr 2023


# airplane automobile bird cat deer dog frog horse ship truck

tar -xvf fmnist_test.tar &> /dev/null

cd ./fmnist_test

cd ankleBoot
mv *.png ../
cd ..
rm -r ankleBoot/

cd bag
mv *.png ../
cd ..
rm -r bag/

cd coat
mv *.png ../
cd ..
rm -r coat/

cd dress
mv *.png ../
cd ..
rm -r dress/

cd pullover
mv *.png ../
cd ..
rm -r pullover/

cd sandal
mv *.png ../
cd ..
rm -r sandal/

cd shirt
mv *.png ../
cd ..
rm -r shirt/

cd sneaker
mv *.png ../
cd ..
rm -r sneaker/

cd top
mv *.png ../
cd ..
rm -r top

cd trouser
mv *.png ../
cd ..
rm -r trouser

cd ..
