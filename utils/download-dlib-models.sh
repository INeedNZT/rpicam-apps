#!/bin/bash
set -e

declare -A networks=(
    # Dlib face landmarks
    ["http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"]="shape_predictor_5_face_landmarks.dat.bz2"
    # Dlib face recognition
    ["http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"]="dlib_face_recognition_resnet_model_v1.dat.bz2"
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

    if [[ "$filename" == *.bz2 ]]; then
        bunzip2 -f "$dir/$filename"
    fi
done