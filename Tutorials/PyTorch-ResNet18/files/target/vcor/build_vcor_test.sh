#!/bin/sh

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT


tar -xvf test.tar &> /dev/null
#mv ./build/dataset/cifar10/test ./test
#rm -r ./build
cd ./test

cd beige
mv *.png ../
cd ..
rm -r beige/

cd black
mv *.png ../
cd ..
rm -r black

cd blue
mv *.png ../
cd ..
rm -r blue/

cd brown
mv *.png ../
cd ..
rm -r brown/

cd gold
mv *.png ../
cd ..
rm -r gold/

cd green
mv *.png ../
cd ..
rm -r green/

cd grey
mv *.png ../
cd ..
rm -r grey/

cd orange
mv *.png ../
cd ..
rm -r orange/

cd pink
mv *.png ../
cd ..
rm -r pink/

cd purple
mv *.png ../
cd ..
rm -r purple

cd red
mv *.png ../
cd ..
rm -r red

cd silver
mv *.png ../
cd ..
rm -r silver

cd tan
mv *.png ../
cd ..
rm -r tan

cd white
mv *.png ../
cd ..
rm -r white

cd yellow
mv *.png ../
cd ..
rm -r yellow


cd ..
