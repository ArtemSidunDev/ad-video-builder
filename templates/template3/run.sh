#!/bin/bash
# This script will run three Python scripts sequentially
folder_path=$1

python3 ./templates/template3/build_videos.py $folder_path
python3 ./templates/template3/add_transcription.py $folder_path


