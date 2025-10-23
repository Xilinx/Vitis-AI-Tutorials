#!/bin/bash
#
# Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
# MIT License

# Date 02 Sep. 2025


# Usage:   ./compare_files.sh dir1 dir2
# Example: ./compare_files.sh /path/to/directory1 /path/to/directory2

# Check if the number of arguments is correct
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory1> <directory2>"
    exit 1
fi

DIR1=$1
DIR2=$2

# Function to compare files
compare_files() {
    local file1="$1"
    local file2="$2"
    
    if [ -f "$file1" ] && [ -f "$file2" ]; then
        diff "$file1" "$file2" > /dev/null
        if [ $? -ne 0 ]; then
            echo "Files differ: $file1 and $file2"
        else
            echo "Files are the same: $file1 and $file2"
        fi
    fi
}

# Export the function for use with find
export -f compare_files

# Use `find` to iterate over files in the first directory
#find "$DIR1" -type f | while read -r file; do #DB: all files
find "$DIR1" -type f \( -name "*.cpp" -o -name "*.c" -o -name "*.h" -o -name "*.hpp" -o -name "*.mk" -o -name "*akefile" -o -name "*.sh" \) | while read -r file; do #DB: only files of type *.c *.cpp *.h *.hpp

    # Get the relative path of the file
    relative_path="${file#$DIR1/}"
    # Construct the corresponding file path in the second directory
    file2="$DIR2/$relative_path"
    
    # Compare the files if the corresponding file exists in the second directory
    compare_files "$file" "$file2"
done

