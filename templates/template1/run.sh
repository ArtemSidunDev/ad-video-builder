#!/bin/bash
# This script will run three Python scripts sequentially
DISPLAY_NUM=$((99 + $RANDOM % 100))

Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 &

XVFB_PID=$!

export DISPLAY=:$DISPLAY_NUM

folder_path=$1

sleep 2

folder_path=$1

python3 ./templates/template1/build_background_video.py $folder_path
python3 ./templates/template1/add_foreground.py $folder_path
python3 ./templates/template1/add_transcription.py $folder_path

if ps -p $XVFB_PID > /dev/null

then
    kill $XVFB_PID
fi