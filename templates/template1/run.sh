#!/bin/bash
# This script will run three Python scripts sequentially

folder_path=$1

python3 ./templates/template1/build_background_video.py $folder_path
python3 ./templates/template1/add_foreground.py $folder_path
python3 ./templates/template1/add_transcription.py $folder_path