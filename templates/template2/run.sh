#!/bin/bash
# This script will run three Python scripts sequentially

folder_path=$1

python3 ./templates/template2/build_videos.py $folder_path
python3 ./templates/template2/add_transcription.py $folder_path

