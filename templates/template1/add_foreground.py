# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 00:26:48 2024

@author: codemaven
"""
import os
import cv2
import copy
import numpy as np
from PIL import Image, ImageOps
from moviepy.editor import VideoFileClip, VideoClip, ImageClip
import argparse

parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
args = parser.parse_args()

folder_path = args.folderPath
# Load your image using PIL
video_fps = 30
video_dest_width = 1216
video_dest_height = 2160

GREEN_COLOR = (15, 250, 74)

temp_folder = os.path.join(folder_path, "temp")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)


def get_position(t):
    left_x = foreground_width//2
    right_x = video_dest_width - foreground_width//2
    y = video_dest_height - foreground_height//2
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


def add_foreground(frame,  t):

    foreground_x, foreground_y = get_position(t)
    back_image = background_clip.get_frame(t)
    front_frame = foreground_clip.get_frame(t)
    # front_height, front_width, _ = front_frame.shape
    
    if foreground_x is None or foreground_y is None:
        return back_image
    
    zoomed_frame = build_zoomed_avatar( t = t, avatar_frame = front_frame, zoom_dest=2, offset_y=0)
    frame = replace_green_background(zoomed_frame, back_image, t, fore_center_x= foreground_x, fore_center_y = foreground_y)
    return frame

def build_zoomed_avatar(t, avatar_frame, zoom_dest = 1.0, offset_y = 0, back_media=None):
    """
    Creates a zoomed avatar video with a replaced green background.
    """
    
    # avatar_frame = avatar_video.get_frame(t)
    
    # Zoom the avatar frame
    image_height, image_width, _ = avatar_frame.shape
    new_width, new_height = int(image_width * zoom_dest), int(image_height * zoom_dest)
    
    if back_media == None:
        back_frame = np.full( ( int( image_height*zoom_dest), int( image_width*zoom_dest), 3), GREEN_COLOR, dtype=np.uint8)

    elif isinstance(back_media, VideoClip):
        back_frame = back_media.get_frame(t)

        back_frame = cv2.resize(back_frame, (int(back_frame.shape[1] * zoom_dest), int(back_frame.shape[0] * zoom_dest)), interpolation=cv2.INTER_LINEAR)
    elif isinstance(back_media, Image.Image):
        back_frame = np.array(back_media)
        back_frame = cv2.resize(back_frame, (int(back_frame.shape[1] * zoom_dest), int(back_frame.shape[0] * zoom_dest)), interpolation=cv2.INTER_LINEAR)
    elif isinstance(back_media, np.ndarray):
        back_frame = back_media
        back_frame = cv2.resize(back_frame, (int(back_frame.shape[1] * zoom_dest), int(back_frame.shape[0] * zoom_dest)), interpolation=cv2.INTER_LINEAR)

    if zoom_dest >= 1:
        resized_image = np.array(Image.fromarray(avatar_frame).resize((new_width, new_height), Image.Resampling.LANCZOS))
    
        back_frame[ int(offset_y*zoom_dest):, :] = resized_image[ 0 : int(new_height - offset_y*offset_y), :]   
    else:
        
        resized_image = np.array( Image.fromarray(avatar_frame).resize((new_width, new_height), Image.Resampling.LANCZOS))
        
        top = ((image_height - new_height) // 2) + offset_y
        left = (image_width - new_width) // 2
        
        top = max(top, 0)
        left = max(left, 0)
        
        if top + new_height > image_height:
            new_height = image_height - top
            resized_image = resized_image[:new_height, :]
        if left + new_width > image_width:
            new_width = image_width - left
            resized_image = resized_image[:, :new_width]

        back_frame[top:top+new_height, left:left+new_width] = resized_image
    return back_frame
    
def replace_green_background( foreground_media, replacement_media, t=None, fore_center_x=video_dest_width//2, fore_center_y=video_dest_height//2):
    """
    Replaces the green background in a frame with another image.
    """
    if isinstance(foreground_media, VideoClip):
        frame = copy.deepcopy( foreground_media.get_frame(t))
    elif isinstance(foreground_media, Image.Image):
        frame = copy.deepcopy( np.array(foreground_media))
    elif isinstance(foreground_media, np.ndarray):
        frame = copy.deepcopy( foreground_media)
    
    if isinstance(replacement_media, VideoClip):
        replacement_frame = replacement_media.get_frame(t)
    elif isinstance(replacement_media, Image.Image):
        replacement_frame = np.array(replacement_media)
    elif isinstance(replacement_media, np.ndarray):
        replacement_frame = replacement_media
    
    rows, cols, _ = replacement_frame.shape
    background = copy.deepcopy(replacement_frame)
    
    x_start = max(fore_center_x - frame.shape[1] // 2, 0)
    y_start = max(fore_center_y - frame.shape[0] // 2, 0)
    x_end = min(fore_center_x + frame.shape[1] // 2, cols)
    y_end = min(fore_center_y + frame.shape[0] // 2, rows)
    
    fg_x_start = max(frame.shape[1] // 2 - fore_center_x, 0)
    fg_y_start = max(frame.shape[0] // 2 - fore_center_y, 0)
    fg_x_end = fg_x_start + (x_end - x_start)
    fg_y_end = fg_y_start + (y_end - y_start)
    
    hsv = cv2.cvtColor(frame[fg_y_start:fg_y_end, fg_x_start:fg_x_end], cv2.COLOR_RGB2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inverse = cv2.bitwise_not(mask)
    
    foreground = cv2.bitwise_and(frame[fg_y_start:fg_y_end, fg_x_start:fg_x_end], frame[fg_y_start:fg_y_end, fg_x_start:fg_x_end], mask=mask_inverse)
    
    background[y_start:y_end, x_start:x_end] = cv2.bitwise_and(background[y_start:y_end, x_start:x_end], background[y_start:y_end, x_start:x_end], mask=mask)
    background[y_start:y_end, x_start:x_end] = cv2.add( foreground.astype(np.uint8), background[y_start:y_end, x_start:x_end].astype(np.uint8))
    return background

def remove_green_background(frame):
    """
    Removes the green background from a given frame and makes it transparent.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask_inverse = cv2.bitwise_not(mask)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)  # Convert to RGBA
    frame[:, :, 3] = mask_inverse  # Set alpha channel based on mask
    
    return frame

background_clip = VideoFileClip(
    os.path.join(
        temp_folder,
        "background_video.mp4"))
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4"))

# Set foreground speacker size to 60% of background video
foreground_width = int( background_clip.size[0] * 0.6)
foreground_height = int( foreground_clip.size[1] * foreground_width / foreground_clip.size[0])

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
