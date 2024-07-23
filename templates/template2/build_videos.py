# -*- coding: utf-8 -*-
"""
Created on Mon May  6 07:01:03 2024

@author: codemaven
"""

import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoClip, VideoFileClip, clips_array, concatenate_videoclips
import subprocess
import os
import argparse

parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
args = parser.parse_args()

folder_path = args.folderPath

# Load your image using PIL
video_fps = 30
video_dest_width = 2160
video_dest_height = 3840
wipe_left_time = 800

speed_factor = 1.04

temp_folder = os.path.join(folder_path, "temp")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    
def add_foreground(frame, t):
    global foreground_clip, foreground_x, foreground_y, foreground_width, foreground_height
    # return frame
    if foreground_x is None or foreground_y is None:
        return frame

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

    foreground_with_mask = np.zeros( frame.shape, dtype=np.uint8)
    foreground_with_mask[foreground_y: foreground_y +
                         foreground_height, foreground_x: foreground_x +
                         foreground_width,] = foreground_frame
    # Invert the mask
    black_mask = np.zeros(
        (frame.shape[0],
         frame.shape[1]),
        dtype=np.uint8)
    black_mask[foreground_y: foreground_y + foreground_height,
               foreground_x: foreground_x + foreground_width] = mask_inverse

    total_mask_inverse = cv2.bitwise_not(black_mask)

    background_with_mask = cv2.bitwise_and(
        frame, frame, mask=total_mask_inverse)
    final_frame = cv2.bitwise_or(foreground_with_mask, background_with_mask)
    return final_frame

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size)/window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]

def prepare_image(image_path, dest_width, dest_height):
    # Load the image and convert it to RGB
    image = Image.open(image_path).convert("RGB")
    original_width, original_height = image.size

    # Calculate the aspect ratio of the image and the destination
    aspect_ratio = original_width / original_height
    dest_ratio = dest_width / dest_height

    # Adjust the image size to match the destination aspect ratio
    if aspect_ratio > dest_ratio:
        new_width = int(original_height * dest_ratio)
        new_height = original_height
    else:
        new_width = original_width
        new_height = int(original_width / dest_ratio)

    # Center the cropped area
    left = (original_width - new_width) / 2
    top = (original_height - new_height) / 2
    right = (original_width + new_width) / 2
    bottom = (original_height + new_height) / 2

    # Crop and resize the image
    cropped_image = image.crop((left, top, right, bottom)).resize((dest_width, dest_height), Image.Resampling.LANCZOS)
    return cropped_image

# Updated zoom_frame function with stabilization
def zoom_frame(t):
    global transition_span, zoom_dest, dest_width, dest_height, zoom_image, image_width, image_height
    
    # Calculate the zoom factor based on the zoom destination
    zoom_factor =  (t / transition_span) - (t / (zoom_dest * transition_span)) + (1 / zoom_dest) if zoom_dest > 1 else 1 - (1 - zoom_dest) * (t / transition_span)
    
    # Apply moving average for stabilization
    zoom_factor_smoothed = moving_average(np.array([zoom_factor]), radius=5)[0]
    
    # Calculate new dimensions
    new_width = int(image_width * zoom_factor_smoothed)
    new_height = int(image_height * zoom_factor_smoothed)
    
    # Center the cropped area
    left = (image_width - new_width) / 2
    top = (image_height - new_height) / 2
    right = left + new_width
    bottom = top + new_height
    
    # Crop and resize the image
    cropped_image = zoom_image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((dest_width, dest_height), Image.Resampling.LANCZOS)
    return np.array(resized_image)

# Function to apply increasing blur over the last 0.4 seconds
def apply_increasing_blur(get_frame, t):
    frame = get_frame(t)
    # Calculate the time remaining until the end of the clip
    time_left = max(0, clip_duration - t)
    if time_left <= blur_duration:
        # Calculate the strength of the blur based on the time left
        blur_strength = int(((blur_duration - time_left) / blur_duration) * 50)  # Adjust the multiplier as needed
        # Apply Gaussian blur using OpenCV
        kernel = int(blur_strength) * 2 + 1
        frame = cv2.GaussianBlur(frame, (kernel, kernel), blur_strength)
    return frame

# Function to apply decreasing blur over the first 0.4 seconds
def apply_decreasing_blur(get_frame, t):
    frame = get_frame(t)
    # Duration of the blur effect in seconds
    time_elapsed = t
    if time_elapsed <= blur_duration:
        # Calculate the strength of the blur based on the time elapsed
        blur_strength = int((1 - (time_elapsed / blur_duration)) * 80)  # Adjust the multiplier as needed
        # Apply Gaussian blur using OpenCV
        kernel = blur_strength * 4 + 1  # Kernel size must be odd
        frame = cv2.GaussianBlur(frame, (kernel, kernel), blur_strength)
    return frame

video_spans = [ 2, 2, 3, 4, 3, 2, 5, 5, 4, 5]
transition_spans = [0.4, 0.4, 0, 0.6, 0.4, 0, 0.6, 0.4, 0.4]
def get_video_timespan( idx):
    if idx == 1:
        return sum(video_spans[:idx-1]), sum(video_spans[:idx]) + transition_spans[0] / 2
    elif idx == len(video_spans):
        return sum(video_spans[:idx-1]) - transition_spans[-1] / 2, sum(video_spans[:idx])
    elif idx > 1 and idx < len(video_spans):
        return sum(video_spans[:idx-1])-transition_spans[idx-2]/2, sum(video_spans[:idx]) + transition_spans[idx-1] / 2
    return 0, 0

def get_transition_span( idx):
    return transition_spans[ idx -1]

def get_video_length( idx):
    if idx == 1:
        return video_spans[0] + transition_spans[0] / 2
    elif idx == len(video_spans):
        return video_spans[-1] + transition_spans[-1] / 2
    elif idx > 1 and idx < len(video_spans):
        return video_spans[idx - 1] + (transition_spans[idx - 2] + transition_spans[idx - 1]) / 2
    return 0
    # return video_spans[idx-1]
    
# =================== GLOBAL Variable Declearation ============================

global transition_span, zoom_dest, dest_width, dest_height, zoom_image
global foreground_clip, foreground_width, foreground_height, foreground_x, foreground_y  


# ================ 1. ZoomOut(Top) / Avatar(Bottom) ===========================
# Time 0-2s

clip_length = get_video_length( idx=1)
clip_start, clip_end = get_video_timespan( idx=1)

# Load the image and convert it to RGB
zoom_image = prepare_image(os.path.join(folder_path, "1.png"), video_dest_width, video_dest_height//2)
image_width, image_height = zoom_image.size

dest_width, dest_height = video_dest_width, video_dest_height//2
transition_span = clip_length
zoom_dest = 1.3

# Create the video clip using the modified zoom_in_frame function
upper_video_clip = VideoClip(lambda t: zoom_frame(t), duration=transition_span)

lower_video_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).subclip(clip_start, clip_end).crop( 0, video_dest_height//6, video_dest_width, video_dest_height//2 + video_dest_height//6)

# Stack the videos vertically
composed_clip = clips_array([[upper_video_clip], [lower_video_clip]])

blur_duration = get_transition_span( idx = 1)
clip_duration = clip_length
blurred_clip = composed_clip.fl(apply_increasing_blur)

blurred_clip.subclip( 0, transition_span).write_videofile(f"{temp_folder}/01.mp4", temp_audiofile=f"{temp_folder}/01.mp3", remove_temp=True, codec="libx264", fps=video_fps)

zoom_image.close()
upper_video_clip.close()
lower_video_clip.close()
composed_clip.close()
blurred_clip.close()

# ================ 2. ZoomIn + Avatar(Left) ===================================
# Time 2-4s

clip_length = get_video_length( idx=2)
clip_start, clip_end = get_video_timespan( idx=2)

# Load the image and convert it to RGB
image = Image.open(os.path.join(folder_path, "2.png")).convert("RGB")

original_width, original_height = image.size

aspect_ratio = original_width / original_height
dest_ratio = video_dest_width / (video_dest_height)

if aspect_ratio > dest_ratio:
    new_width = int(original_height * dest_ratio)
    new_height = original_height
else:
    new_width = original_width
    new_height = int(original_width / dest_ratio)

left = (original_width - new_width) / 2
top = (original_height - new_height) / 2
right = (original_width + new_width) / 2
bottom = (original_height + new_height) / 2

zoom_image = image.crop((left, top, right, bottom)).resize((video_dest_width, video_dest_height))
image_width, image_height = zoom_image.size

dest_width, dest_height = video_dest_width, video_dest_height
transition_span = clip_length
zoom_dest = 0.9

# Create the video clip using the modified zoom_in_frame function
background_clip = VideoClip(lambda t: zoom_frame(t), duration=transition_span)
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip( clip_start, clip_end)

# Set foreground speacker size to 60% of background video
foreground_width = int(background_clip.size[0] * 0.6)
foreground_height = int( foreground_clip.size[1] * foreground_width / foreground_clip.size[0])
foreground_clip = foreground_clip.resize((foreground_width, foreground_height))

foreground_x, foreground_y = 0, video_dest_height - foreground_height
# Apply the replace_green_background function to each frame of the
# Deley for 2 seconds
processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t))

blur_duration = get_transition_span( idx = 1)
blurred_clip = processed_clip.fl(apply_decreasing_blur)
# Write the processed video to a file
blurred_clip.subclip( 0, transition_span).write_videofile( f"{temp_folder}/02.mp4", codec="libx264", fps=video_fps)

zoom_image.close()
background_clip.close()
foreground_clip.close()
processed_clip.close()
blurred_clip.close()

# ================ 3. ScreenCast + Avatar(Center) =============================
# Time 4-7s
# video starts from 5s
clip_length = get_video_length( idx=3)
clip_start, clip_end = get_video_timespan( idx=3)

clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).subclip(5, 5 + clip_length)
aspect_ratio = clip.w / clip.h
dest_ratio = video_dest_width / video_dest_height

if aspect_ratio > dest_ratio:
    new_width = int(clip.h * dest_ratio)
    new_height = clip.h
else:
    new_width = clip.w
    new_height = int(clip.w / dest_ratio)

cropped_clip = clip.crop( 0,  102,  new_width, 102 + new_height).resize( (video_dest_width, video_dest_height))

background_clip = cropped_clip.subclip( 0, clip_length)

foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip( clip_start, clip_end).crop(
    0, int( video_dest_height / (0.8 * 8)), 0, int( 5*video_dest_height/ (0.8 * 8))).resize((video_dest_width*0.8, video_dest_height//2))

# foreground_clip.write_videofile("output.mp4", fps=video_fps, codec="libx264")
foreground_width, foreground_height = foreground_clip.size[0], foreground_clip.size[1]
foreground_x, foreground_y = ( video_dest_width - foreground_width)//2, video_dest_height // 2

processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t))

# Write the processed video to a file
processed_clip.subclip(0, clip_length).write_videofile( f"{temp_folder}/03.mp4", codec="libx264", fps=video_fps)

clip.close()
cropped_clip.close()
foreground_clip.close()
background_clip.close()
processed_clip.close()

# ================ 4. Moving left + Avatar( Top-Center) =======================
# Time 7-9s, 9-11s

clip_length = get_video_length( idx=4)
clip_start, clip_end = get_video_timespan( idx=4)

clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).subclip(8, 5 + clip_length)
aspect_ratio = clip.w / clip.h
dest_ratio = video_dest_width / video_dest_height

if aspect_ratio > dest_ratio:
    new_width = int(clip.h * dest_ratio)
    new_height = clip.h
else:
    new_width = clip.w
    new_height = int(clip.w / dest_ratio)

upper_video_clip = clip.crop( 0,  102,  new_width, 102 + new_height).resize( (video_dest_width, video_dest_height))

image = Image.open(os.path.join(folder_path, "3.png"))
original_width, original_height = image.size

aspect_ratio = original_width / original_height
dest_ratio = 1.5

if aspect_ratio > dest_ratio:
    new_width = int(original_height * dest_ratio)
    new_height = original_height
else:
    new_width = original_width
    new_height = int(original_width / dest_ratio)

left = (original_width - new_width) / 2
top = (original_height - new_height) / 2
right = (original_width + new_width) / 2
bottom = (original_height + new_height) / 2

move_image = image.crop((left, top, right, bottom)).resize((video_dest_height*3//4, video_dest_height // 2))

image_width, image_height = move_image.size

def moving_left_frame(t):
    new_left = int((image_width - video_dest_width) * ( 1 - t / duration))  # Moves from right to left
    cropped_image = move_image.crop(
        (image_width - new_left - video_dest_width, 0, image_width - new_left, video_dest_height//2))
    return np.array(cropped_image)

duration = clip_length/2 + 0.2
lower_video_clip1 = VideoClip(moving_left_frame, duration=duration)

blur_duration = 0.4
clip_duration = duration
lower_blurred_clip1 = lower_video_clip1.fl(apply_increasing_blur)
lower_blurred_clip1.write_videofile(f"{temp_folder}/04-1.mp4", codec="libx264", fps=video_fps)

image.close()
move_image.close()


image = Image.open(os.path.join(folder_path, "4.png"))
original_width, original_height = image.size

aspect_ratio = original_width / original_height
dest_ratio = 1.5

if aspect_ratio > dest_ratio:
    new_width = int(original_height * dest_ratio)
    new_height = original_height
else:
    new_width = original_width
    new_height = int(original_width / dest_ratio)

left = (original_width - new_width) / 2
top = 0
right = (original_width + new_width) / 2
bottom = new_height

move_image = image.crop((left, top, right, bottom)).resize((video_dest_height*3//4, video_dest_height // 2))

image_width, image_height = move_image.size

def moving_right_frame(t):
    new_left = int((image_width - video_dest_width) * t / duration)  # Moves from right to left
    cropped_image = move_image.crop(
        (image_width - new_left - video_dest_width, 0, image_width - new_left, video_dest_height//2))
    return np.array(cropped_image)

duration = clip_length/2 + 0.2
lower_video_clip2 = VideoClip(moving_right_frame, duration=duration)

blur_duration = 0.4
clip_duration = duration
lower_blurred_clip2 = lower_video_clip2.fl(apply_decreasing_blur)
lower_blurred_clip2.write_videofile(f"{temp_folder}/04-2.mp4", codec="libx264", fps=video_fps)

output_file = f"{temp_folder}/04_lower.mp4"

command = "ffmpeg-concat -T ./templates/template2/input/zoomin_transition.json -o " + output_file + f" {temp_folder}/04-1.mp4" + f" {temp_folder}/04-2.mp4"
print(command)
try:
    completed_process = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True
    )
    if completed_process.returncode == 0:
        print("Command output:")
        print(completed_process.stdout)
        print("Command executed successfully.")
    else:
        print(f"command includes errors:  {completed_process.stderr}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e.stderr}")

lower_video_clip = VideoFileClip(f"{temp_folder}/04_lower.mp4").subclip( 0, clip_length)

# add trasition
def pop_from_bottom(t):
    if t <= trans_duration:
        new_height = int(video_dest_height * t // (2 * trans_duration))
    else:
        new_height = video_dest_height // 2 
    global foreground_x, foreground_y, foreground_width, foreground_height
    foreground_width, foreground_height = foreground_clip.size[0], foreground_clip.size[1]
    foreground_x = ( video_dest_width - foreground_width)//2
    foreground_y = video_dest_height // 2 - new_height    
    lower_frame = lower_video_clip.get_frame(t)
    upper_frame = upper_video_clip.get_frame(t)
        # Crop the lower frame using slicing
    cropped_image = lower_frame[:new_height, :video_dest_width]
    
    # Paste the cropped image onto the upper frame
    upper_frame[video_dest_height - new_height:, :video_dest_width] = cropped_image
    return add_foreground( upper_frame, t)
    # return upper_frame

trans_duration = 1/4
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip( clip_start ,clip_end).crop(
    0, int( video_dest_height / (0.8 * 8)), 0, int( 5*video_dest_height/ (0.8 * 8))).resize((video_dest_width*0.8, video_dest_height//2))

processed_clip = VideoClip(pop_from_bottom, duration=4)

blur_duration = get_transition_span( idx = 4)
clip_duration = clip_length
blurred_clip = processed_clip.fl(apply_increasing_blur)

blurred_clip.subclip( 0, clip_length).write_videofile( f"{temp_folder}/04.mp4", codec="libx264", fps=video_fps)

image.close()
cropped_clip.close()
upper_video_clip.close()
lower_video_clip1.close()
lower_video_clip2.close()
lower_blurred_clip1.close()
lower_blurred_clip2.close()
lower_video_clip.close()
processed_clip.close()
blurred_clip.close()

# ================ 5. ZoomIn + Avatar( Left) ==================================
# Time 11-14s

clip_length = get_video_length( idx=5)
clip_start, clip_end = get_video_timespan( idx=5)

# zoom_image = Image.open(os.path.join(folder_path, "5.png")).convert("RGB")
zoom_image = prepare_image(os.path.join(folder_path, "5.png"), video_dest_width, video_dest_height)
image_width, image_height = zoom_image.size

dest_width, dest_height = video_dest_width, video_dest_height

transition_span = clip_length
zoom_dest = 0.9

# Create the video clip using the modified zoom_in_frame function
background_clip = VideoClip(lambda t: zoom_frame(t), duration=transition_span)

foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip( clip_start, clip_end)

# Set foreground speacker size to 60% of background video
foreground_width = int(background_clip.size[0] * 0.6)
foreground_height = int( foreground_clip.size[1] * foreground_width / foreground_clip.size[0])
foreground_clip = foreground_clip.resize((foreground_width, foreground_height))

foreground_x, foreground_y = 0, video_dest_height - foreground_height
# Apply the replace_green_background function to each frame of the
processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t))

# Write the processed video to a file
# processed_clip.subclip(0, clip_length).write_videofile( f"{temp_folder}/05.mp4", codec="libx264", fps=video_fps)

blur_duration = get_transition_span( idx = 4)
clip_duration = clip_length
blurred_clip = processed_clip.fl(apply_decreasing_blur)

blurred_clip.subclip( 0, clip_length).write_videofile( f"{temp_folder}/05.mp4", codec="libx264", fps=video_fps)

zoom_image.close()
background_clip.close()
foreground_clip.close()
processed_clip.close()
blurred_clip.close()

# ================ 6. Full screen avatar video ================================
# Time 14-16s

clip_length = get_video_length( idx=6)
clip_start, clip_end = get_video_timespan( idx=6)

avatar_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).subclip(clip_start ,clip_end)
avatar_clip.write_videofile(f"{temp_folder}/06.mp4", temp_audiofile=f"{temp_folder}/06.mp3", remove_temp=True, codec="libx264", fps=video_fps)
avatar_clip.close()

# ================ 7. Full screen avatar video ================================
# Time 16-18.5s, 18.5-21

clip_length = get_video_length( idx=7)
clip_start, clip_end = get_video_timespan( idx=7)

image = Image.open(os.path.join(folder_path, "6.png")).convert("RGB")

original_width, original_height = image.size

aspect_ratio = original_width / original_height
dest_ratio = video_dest_width / (video_dest_height //2)

if aspect_ratio > dest_ratio:
    new_width = int(original_height * dest_ratio)
    new_height = original_height
else:
    new_width = original_width
    new_height = int(original_width / dest_ratio)

left = (original_width - new_width) / 2
top = (original_height - new_height) / 2
right = (original_width + new_width) / 2
bottom = (original_height + new_height) / 2

zoom_image = image.crop((left, top, right, bottom)).resize((video_dest_width, video_dest_height // 2))
image_width, image_height = zoom_image.size

dest_width, dest_height = video_dest_width, video_dest_height//2
transition_span = clip_length/2 + 0.2
zoom_dest = 0.9
# Create the video clip using the modified zoom_in_frame function
upper_video_clip1 = VideoClip(lambda t: zoom_frame(t), duration=transition_span)

blur_duration = 0.4
clip_duration = transition_span
upper_blurred_clip1 = upper_video_clip1.fl(apply_increasing_blur)

upper_blurred_clip1.write_videofile(f"{temp_folder}/07-1.mp4", fps=video_fps)

image.close()
zoom_image.close()

# for 7_2
image = Image.open(os.path.join(folder_path, "7.png")).convert("RGB")

original_width, original_height = image.size

aspect_ratio = original_width / original_height
dest_ratio = video_dest_width / (video_dest_height //2)

#zoomin 125%
if aspect_ratio > dest_ratio:
    new_width = int( original_height * dest_ratio * 0.8)
    new_height = int( original_height * 0.8)
else:
    new_width = int( original_width * 0.8)
    new_height = int( original_width * 0.8 / dest_ratio)

left = (original_width - new_width) / 2
top = (original_height - new_height) / 2
right = (original_width + new_width) / 2
bottom = (original_height + new_height) / 2

zoom_image = image.crop((left, top, right, bottom)).resize((video_dest_width, video_dest_height // 2))
image_width, image_height = zoom_image.size

dest_width, dest_height = video_dest_width, video_dest_height//2
transition_span = clip_length/2 + 0.2

zoom_dest = 1.1
upper_video_clip2 = VideoClip(lambda t: zoom_frame(t), duration=transition_span)

blur_duration = 0.4
clip_duration = transition_span
upper_blurred_clip2 = upper_video_clip2.fl(apply_decreasing_blur)

upper_blurred_clip2.write_videofile(f"{temp_folder}/07-2.mp4", fps=video_fps)

output_file = f"{temp_folder}/07_upper.mp4"

command = "ffmpeg-concat -T ./templates/template2/input/zoomin_transition.json -o " + output_file +  f" {temp_folder}/07-1.mp4" + f" {temp_folder}/07-2.mp4"

try:
    completed_process = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True
    )
    if completed_process.returncode == 0:
        print("Command output:")
        print(completed_process.stdout)
        print("Command executed successfully.")
    else:
        print(f"command includes errors:  {completed_process.stderr}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e.stderr}")

upper_video_clip = VideoFileClip(f"{temp_folder}/07_upper.mp4").subclip( 0, clip_length)

lower_video_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).subclip( clip_start, clip_end).crop( 0, video_dest_height//6, video_dest_width, video_dest_height//2 + video_dest_height//6)

# Stack the videos vertically
final_clip = clips_array([[upper_video_clip], [lower_video_clip]])
# final_clip.write_videofile(f"{temp_folder}/08.mp4", codec="libx264")

# Add transition
def drop_from_top(t):
    if t <= transition_span:
        new_height = int(video_dest_height * t // (2 * transition_span))
        nu_height = int(video_dest_height * t // (3 * transition_span)) + video_dest_height//6
    else:
        new_height = video_dest_height // 2
        nu_height = video_dest_height // 2
    lower_frame = lower_video_clip.get_frame(t)
    upper_frame = upper_video_clip.get_frame(t)
    rest_frame = back_frame_clip.get_frame(t)

    back_frame[ :new_height, :video_dest_width] = upper_frame[video_dest_height//2-new_height:video_dest_height//2, :video_dest_width]
    back_frame[ nu_height:nu_height + video_dest_height//2,] = lower_frame
    back_frame[ new_height:nu_height,] = rest_frame[video_dest_height//6-(nu_height - new_height):video_dest_height//6]
    back_frame[ nu_height + video_dest_height//2:,] = rest_frame[video_dest_height*2//3:video_dest_height*7//6-nu_height]
    
    return back_frame

back_frame = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).get_frame(clip_start).copy()
print(back_frame.shape)
transition_span = 1/4

back_frame_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).subclip( clip_start, clip_start + transition_span)
processed_clip = VideoClip(drop_from_top, duration=5)

blur_duration = get_transition_span( idx = 7)
clip_duration = clip_length
blurred_clip = processed_clip.fl(apply_increasing_blur)

blurred_clip.subclip(0, clip_length).write_videofile(f"{temp_folder}/07.mp4", codec="libx264", fps=video_fps)

image.close()
zoom_image.close()
upper_video_clip1.close()
upper_video_clip2.close()
upper_blurred_clip1.close()
upper_blurred_clip2.close()
upper_video_clip.close()
lower_video_clip.close()
blurred_clip.close()

# ================ 8. ZoomIn + avatar(right) =================================
# Time 21-26s

clip_length = get_video_length( idx=8)
clip_start, clip_end = get_video_timespan( idx=8)

# Load the image and convert it to RGB
image = Image.open(os.path.join(folder_path, "8.png")).convert("RGB")

original_width, original_height = image.size

aspect_ratio = original_width / original_height
dest_ratio = video_dest_width / (video_dest_height)

if aspect_ratio > dest_ratio:
    new_width = int(original_height * dest_ratio)
    new_height = original_height
else:
    new_width = original_width
    new_height = int(original_width / dest_ratio)

left = (original_width - new_width) / 2
top = (original_height - new_height) / 2
right = (original_width + new_width) / 2
bottom = (original_height + new_height) / 2

# global transition_span, zoom_dest, dest_width, dest_height, zoom_image
zoom_image = image.crop((left, top, right, bottom)).resize((video_dest_width, video_dest_height))
image_width, image_height = zoom_image.size

dest_width, dest_height = video_dest_width, video_dest_height
transition_span = clip_length
zoom_dest = 0.9

# Create the video clip using the modified zoom_in_frame function
background_clip = VideoClip(lambda t: zoom_frame(t), duration=transition_span)
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).speedx(speed_factor).subclip( clip_start, clip_end)

# Set foreground speacker size to 60% of background video
foreground_width = int(background_clip.size[0] * 0.6)
foreground_height = int( foreground_clip.size[1] * foreground_width / foreground_clip.size[0])
foreground_clip = foreground_clip.resize((foreground_width, foreground_height))

foreground_x, foreground_y = video_dest_width - foreground_width, video_dest_height - foreground_height
# Apply the replace_green_background function to each frame of the
processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t))

blur_duration = get_transition_span( idx = 7)
blurred_clip = processed_clip.fl(apply_decreasing_blur)

# Write the processed video to a file
blurred_clip.subclip( 0, clip_length).write_videofile( f"{temp_folder}/08.mp4", codec="libx264", fps=video_fps)

image.close()
background_clip.close()
foreground_clip.close()
processed_clip.close()
blurred_clip.close()

# ================ 09. Full screen avatar video ===============================
# Time 26-30s

clip_length = get_video_length( idx=9)
clip_start, clip_end = get_video_timespan( idx=9)

clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4"))
clip_end = clip.duration

avatar_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).subclip( clip_start, clip_end)
avatar_clip.write_videofile(f"{temp_folder}/09.mp4", temp_audiofile=f"{temp_folder}/09.mp3", remove_temp=True, codec="libx264", fps=video_fps)
avatar_clip.close()

# ================ 10. Blur video ===============================
# Time 30-35s

clip_length = get_video_length( idx=10)
clip_start, clip_end = get_video_timespan( idx=10)
action_delay = 1.5

def blur_frame(frame):
    return cv2.GaussianBlur(frame, (51, 51), 10)

blur_amount = 3

clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).subclip(10, 10+clip_length)
aspect_ratio = clip.w / clip.h
dest_ratio = video_dest_width / video_dest_height

if aspect_ratio > dest_ratio:
    new_width = int(clip.h * dest_ratio)
    new_height = clip.h
else:
    new_width = clip.w
    new_height = int(clip.w / dest_ratio)

cropped_clip = clip.crop( 0,  102,  new_width, 102+new_height).resize( (video_dest_width, video_dest_height))
background_clip = cropped_clip.fl_image(blur_frame)

foreground_clip = VideoFileClip(os.path.join(folder_path, "action.mp4")).subclip( 0, clip_length - action_delay)

foreground_width = background_clip.size[0]
foreground_height = int( foreground_clip.size[1] * foreground_width / foreground_clip.size[0])
foreground_clip = foreground_clip.resize((foreground_width, foreground_height))

foreground_x = 0
foreground_y = (background_clip.size[1] - foreground_height) // 2

processed_clip =  concatenate_videoclips([ background_clip.subclip( 0, action_delay)
                                          , background_clip.subclip( action_delay, background_clip.duration).fl(lambda gf, t: add_foreground(gf(t), t))])

# Write the processed video to a file
processed_clip.subclip(0, clip_length).write_videofile( f"{temp_folder}/10.mp4", codec="libx264", fps=video_fps)

background_clip.close()
foreground_clip.close()
processed_clip.close()

# =============================================================================
# =========================  Concantenate Videos  =============================
# =============================================================================

video9 = VideoFileClip(f"{temp_folder}/09.mp4")

commands = [
    f'ffmpeg -i 02.mp4 -i 03.mp4 -filter_complex "[0][1]xfade=transition=slideup:duration={get_transition_span(idx=2)}:offset={video_spans[1]},format=yuv420p" 02-03.mp4',
    f'ffmpeg -i 05.mp4 -i 06.mp4 -filter_complex "[0][1]xfade=transition=slideleft:duration={get_transition_span(idx=5)}:offset={video_spans[4]},format=yuv420p" 05-06.mp4',
    f'ffmpeg -i 09.mp4 -i 10.mp4 -filter_complex "[0][1]xfade=transition=slideleft:duration={get_transition_span(idx=9)}:offset={video9.duration - get_transition_span(idx=9)},format=yuv420p" 09-10.mp4',
]

# Execute each command in the temp folder
for command in commands:
    try:
        completed_process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            cwd=temp_folder
        )
        if completed_process.returncode == 0:
            print("Command output:")
            print(completed_process.stdout)
            print("Command executed successfully.")
        else:
            print(f"command includes errors:  {completed_process.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e.stderr}")


video_clip_names = ['01.mp4', '02-03.mp4', '04.mp4', '05-06.mp4', '07.mp4', '08.mp4', '09-10.mp4']

#for each video clip, add the path to the temp folder
video_clip_names = [os.path.join(temp_folder, clip) for clip in video_clip_names]
background_video = os.path.join(temp_folder, 'background_video.mp4')

command = ['ffmpeg-concat', '-T', "./templates/template2/input/transition.json",
           '-o', background_video] + video_clip_names
try:
    completed_process = subprocess.run(
        " ".join(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True)
    if completed_process.returncode == 0:
        print("Command output:")
        print(completed_process.stdout)
        print("Command executed successfully.")

    else:
        print(f"command includes errors:  {completed_process.stderr}")
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")


for file_name in os.listdir(temp_folder):
    file_path = os.path.join(temp_folder, file_name)

    if file_name != "background_video.mp4":
        os.remove(file_path)
