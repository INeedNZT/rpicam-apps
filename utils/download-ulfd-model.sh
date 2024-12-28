#!/bin/bash
set -e

declare -A networks=(
    # tflite RFB version model
    ["https://raw.githubusercontent.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/master/tflite/pretrained/version-RFB-320_without_postprocessing.tflite"]="version-RFB-320_without_postprocessing.tflite"
)

if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

dir=$1
mkdir -p $dir

for url in "${!networks[@]}"; do
    filename="${networks[$url]}"
    wget -nv -O "$dir/$filename" "$url"
done
