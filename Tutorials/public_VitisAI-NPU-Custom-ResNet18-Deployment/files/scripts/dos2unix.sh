#!/bin/bash

# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# Date 02 Sep. 2025

for file in $(find . -name "*.*"); do
    echo ${file}
    dos2unix ${file}
done
