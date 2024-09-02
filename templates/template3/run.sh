#!/bin/bash
# This script will run three Python scripts sequentially

DISPLAY_NUM=$((99 + $RANDOM % 100))

# Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 &

XVFB_PID=$!

export DISPLAY=:$DISPLAY_NUM

folder_path=$1
subtitle_template=$2

sleep 2

python3 ./templates/template3/build_videos.py $folder_path $subtitle_template
python3 ./templates/template3/add_transcription.py $folder_path $subtitle_template

if ps -p $XVFB_PID > /dev/null

then
    kill $XVFB_PID
fi


