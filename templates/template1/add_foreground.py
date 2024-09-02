# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 00:26:48 2024

@author: codemaven
"""

import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import os
import argparse

parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
parser.add_argument('subtitleTemplate', type=str, nargs='?', default=None, help='Select the subtitle template')
args = parser.parse_args()

folder_path = args.folderPath
# Load your image using PIL
video_fps = 30
video_dest_width = 1216
video_dest_height = 2160

temp_folder = os.path.join(folder_path, "temp")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)


def get_position(t):
    left_x = 0
    right_x = video_dest_width - foreground_width
    y = video_dest_height - foreground_height
    if t <= 3:
        x = left_x
    elif t <= 5:
        x = right_x
    elif t <= 7:
        x = left_x
    elif t <= 10:
        x = right_x
    elif t <= 16.5:
        x = left_x
    elif t <= 19.5:
        x = None
        y = None
    elif t <= 21:
        x = left_x
    elif t <= 24:
        x = right_x
    else:
        x = None
        y = None
    return x, y


def add_foreground(frame, t):

    foreground_x, foreground_y = get_position(t)

    # return frame
    if foreground_x is None or foreground_y is None:
        return background_clip.get_frame(t)

    front_frame = foreground_clip.get_frame(t)
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(front_frame, cv2.COLOR_RGB2HSV)

    # Define lower and upper bounds for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask_inverse = cv2.bitwise_not(mask)
    foreground_frame = cv2.bitwise_and(
        front_frame, front_frame, mask=mask_inverse)

    foreground_with_mask = np.zeros(
        (background_clip.size[1], background_clip.size[0], 3), dtype=np.uint8)
    foreground_with_mask[foreground_y: foreground_y +
                         foreground_height, foreground_x: foreground_x +
                         foreground_width,] = foreground_frame
    # Invert the mask
    black_mask = np.zeros(
        (background_clip.size[1],
         background_clip.size[0]),
        dtype=np.uint8)
    black_mask[foreground_y: foreground_y + foreground_height,
               foreground_x: foreground_x + foreground_width] = mask_inverse

    total_mask_inverse = cv2.bitwise_not(black_mask)

    background_with_mask = cv2.bitwise_and(
        frame, frame, mask=total_mask_inverse)
    final_frame = cv2.bitwise_or(foreground_with_mask, background_with_mask)
    return final_frame


background_clip = VideoFileClip(
    os.path.join(
        temp_folder,
        "background_video.mp4"))
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4"))

# Set foreground speacker size to 60% of background video
foreground_width = int(background_clip.size[0] * 0.6)
foreground_height = int(
    foreground_clip.size[1] *
    foreground_width /
    foreground_clip.size[0])
foreground_clip = foreground_clip.resize((foreground_width, foreground_height))

# Apply the replace_green_background function to each frame of the
# foreground video
processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t))

# Write the processed video to a file
processed_clip.write_videofile(
    os.path.join(
        temp_folder,
        "overlayed_video.mp4"),
    fps=30)
