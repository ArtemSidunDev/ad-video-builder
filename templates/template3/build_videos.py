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
video_dest_width = 1080
video_dest_height = 1920
wipe_left_time = 400

temp_folder = os.path.join(folder_path, "temp")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    
# Create a video with increasing size and decreasing transparency
def appearance_effect(t):
    global transition_span
    back_frame = background_clip.get_frame(t).copy()
    fore_frame = foreground_clip.get_frame(t)
    
    fore_width, fore_height =  foreground_clip.w, foreground_clip.h

    if t < transition_span:
        fore_width = fore_width // 2 + int( fore_width // 2 * t/transition_span)
        fore_height = fore_height // 2 + int( fore_height // 2 * t/transition_span)
    elif t > background_clip.duration - transition_span:
        fore_width = fore_width // 2 + int( fore_width // 2 * (background_clip.duration-t)/transition_span)
        fore_height = fore_height // 2 + int( fore_height // 2 * (background_clip.duration-t)/transition_span)
        
    left = (background_clip.w - fore_width) // 2 + left_offest
    top = int(background_clip.h * 3 / 4 - fore_height/2)
    right = left + fore_width
    bottom = top + fore_height
    
    if t < transition_span:
        back_frame[top:bottom, left:right]= back_frame[top:bottom, left:right] * (1-t/transition_span) + cv2.resize( fore_frame, (right-left, bottom-top)) * (t/transition_span)
    elif t > background_clip.duration - transition_span:
        back_frame[top:bottom, left:right]= back_frame[top:bottom, left:right] * (1-(background_clip.duration-t)/transition_span) + cv2.resize( fore_frame, (right-left, bottom-top)) * ( ( background_clip.duration-t)/transition_span)
    else:
        back_frame[top:bottom, left:right]= fore_frame
    return back_frame

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

# Updated zoom_frame function with stabilization
def zoom_frame(t):
    global transition_span, zoom_dest, dest_width, dest_height, zoom_image, image_width, image_height
    
    # Calculate the zoom factor based on the zoom destination
    zoom_factor =  (t / transition_span) - (t / (zoom_dest * transition_span)) + (1 / zoom_dest) if zoom_dest > 1 else 1 - (1 - zoom_dest) * (t / transition_span)
    
    # Apply moving average for stabilization
    zoom_factor_smoothed = moving_average(np.array([zoom_factor]), radius=20)[0]
    
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

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size)/window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]

# Updated zoom_frame function with stabilization
def zoom_and_move_frame(t):
    global transition_span, zoom_dest, dest_width, dest_height, image_width, image_height, is_to_right
    
    new_left = int((image_width - video_dest_width) * t / transition_span) # Moves from right to left
    if is_to_right:
        new_left = image_width - video_dest_width - new_left
    
    zoom_image = src_image.crop( ( new_left, 0, new_left + dest_width, dest_height))
    croped_width, croped_height = zoom_image.size
    
    # Calculate the zoom factor based on the zoom destination
    zoom_factor =  (t / transition_span) - (t / (zoom_dest * transition_span)) + (1 / zoom_dest) if zoom_dest > 1 else 1 - (1 - zoom_dest) * (t / transition_span)
    
    # Apply moving average for stabilization
    zoom_factor_smoothed = moving_average(np.array([zoom_factor]), radius=20)[0]
    
    # Calculate new dimensions
    new_width = int(croped_width * zoom_factor_smoothed)
    new_height = int(croped_height * zoom_factor_smoothed)
    
    # Center the cropped area
    left = (croped_width - new_width) / 2
    top = (croped_height - new_height) / 2
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
        blur_strength = int(((blur_duration - time_left) / blur_duration) * 80)  # Adjust the multiplier as needed
        # Apply Gaussian blur using OpenCV
        kernel = int(blur_strength) * 4 + 1
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

video_spans = [ 2, 2, 3, 4, 4, 3, 3, 3, 6.7, 4.3]
transition_spans = [0.6, 0.6, 0, 0.4, 0.6, 0.6, 0.4, 0.4, 0.6]
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

# =================== GLOBAL Variable Declearation ============================
global transition_span, zoom_dest, dest_width, dest_height, image_width, image_height, is_to_right, foreground_x, foreground_y
# ===================== 1. Zoom and Moving Left ===============================
# Time: 0-2s

clip_length = get_video_length( idx=1)
clip_start, clip_end = get_video_timespan( idx=1)

image = Image.open(os.path.join(folder_path, "1.png")).convert("RGB")
original_width, original_height = image.size
src_image = image.resize( ( int( original_width * video_dest_height / original_height) , video_dest_height))

image_width, image_height = src_image.size
zoom_dest = 0.9
transition_span = clip_length
dest_width, dest_height = video_dest_width, video_dest_height
is_to_right = False

processed_clip = VideoClip(zoom_and_move_frame, duration=clip_length)

blur_duration = get_transition_span( idx = 1)
clip_duration = clip_length
blurred_clip = processed_clip.fl(apply_increasing_blur)

blurred_clip.subclip( 0, clip_length).write_videofile(f"{temp_folder}/01.mp4", codec="libx264", fps=video_fps)

image.close()
src_image.close()
processed_clip.close()
blurred_clip.close()

# ===================== 2. Zoom and Moving Right ==============================
# Time: 2-4s

clip_length = get_video_length( idx=2)
clip_start, clip_end = get_video_timespan( idx=2)

image = Image.open(os.path.join(folder_path, "2.png")).convert("RGB")
original_width, original_height = image.size
src_image = image.resize( ( int( original_width * video_dest_height / original_height) , video_dest_height))

image_width, image_height = src_image.size
zoom_dest = 0.9
transition_span = clip_length
dest_width, dest_height = video_dest_width, video_dest_height
is_to_right = True

video_clip = VideoClip(zoom_and_move_frame, duration = clip_length)

blur_duration = get_transition_span( idx = 1)
blurred_clip = video_clip.fl(apply_decreasing_blur)
# Write the processed video to a file
blurred_clip.subclip( 0, clip_length).write_videofile( f"{temp_folder}/02.mp4", codec="libx264", fps=video_fps)

image.close()
src_image.close()
video_clip.close()
blurred_clip.close()

# Concatenate video 1 and video 2
command = 'xvfb-run -s "-screen 0 1024x768x24" ' + f"ffmpeg-concat -t SimpleZoom -d {int(transition_spans[0]*1000)} -o 01-02-back.mp4 01.mp4 02.mp4"

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
    
background_clip = VideoFileClip(f"{temp_folder}/01-02-back.mp4")
    
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip( 0, background_clip.duration).resize((video_dest_width*0.75, video_dest_height*0.75))

foreground_width, foreground_height = foreground_clip.size[0], foreground_clip.size[1]
foreground_x, foreground_y = ( video_dest_width - foreground_width)//2, video_dest_height - foreground_height

processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t))

blur_duration = get_transition_span( idx = 2)
clip_duration = processed_clip.duration
blurred_clip = processed_clip.fl(apply_increasing_blur)

blurred_clip.write_videofile( f"{temp_folder}/01-02.mp4", codec="libx264", fps=video_fps)

foreground_clip.close()
background_clip.close() 
processed_clip.close()
blurred_clip.close()
# ===================== 3. Smoothly Emerge and Disappear ======================
# Time: 4-7s

clip_length = get_video_length( idx=3)
clip_start, clip_end = get_video_timespan( idx=3)

background_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip(clip_start, clip_end)
foreground_clip_orignal = VideoFileClip(os.path.join(folder_path, "ss.mp4")).subclip(0, clip_length)
# Resize foreground video to match background video dimensions
foreground_clip_cropped = foreground_clip_orignal.crop( 0, 110, foreground_clip_orignal.w, foreground_clip_orignal.h - 200)
foreground_clip = foreground_clip_cropped.resize( ( background_clip.w * 0.4, background_clip.w * 0.4 / foreground_clip_cropped.w * foreground_clip_cropped.h))

# Set paramters for appearance_effect
transition_span = 0.3
left_offest = 0

# Apply effect
processed_clip = VideoClip(appearance_effect, duration=clip_length)

blur_duration = get_transition_span( idx = 2)
blurred_clip = processed_clip.fl(apply_decreasing_blur)

# Write the final video to a file
blurred_clip.write_videofile(f"{temp_folder}/03.mp4", codec="libx264", fps=video_fps)

background_clip.close()
foreground_clip_orignal.close()
foreground_clip_cropped.close()
foreground_clip.close()
processed_clip.close()
blurred_clip.close()

# ===================== 4. Smoothly Emerge and Disappear Zoomming In 2 Videos ====
# Time: 7-11s

clip_length = get_video_length( idx=4)
clip_start, clip_end = get_video_timespan( idx=4)

background_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip(clip_start, clip_end)

image = Image.open(os.path.join(folder_path, "3.png")).convert("RGB")

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

dest_width, dest_height = video_dest_width * 2 // 5, video_dest_height * 2 // 5
transition_span = clip_length
zoom_dest = 0.7
# Create the video clip using the modified zoom_in_frame function
fore_clip1 = VideoClip(lambda t: zoom_frame(t), duration=clip_length)
fore_clip1.write_videofile(f"{temp_folder}/04-1.mp4", codec="libx264", fps=video_fps)

image.close()
zoom_image.close()
fore_clip1.close()

image = Image.open(os.path.join(folder_path, "4.png")).convert("RGB")

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

dest_width, dest_height = video_dest_width * 2 // 5, video_dest_height * 2 // 5
transition_span = clip_length
zoom_dest = 0.7
# Create the video clip using the modified zoom_in_frame function
fore_clip2 = VideoClip(lambda t: zoom_frame(t), duration=clip_length)
fore_clip2.write_videofile(f"{temp_folder}/04-2.mp4", codec="libx264", fps=video_fps)

image.close()
zoom_image.close()
fore_clip2.close()

foreground_clip = VideoFileClip(f"{temp_folder}/04-1.mp4")

transition_span = 0.3
left_offest = -video_dest_width // 4
processed_clip = VideoClip(appearance_effect, duration=clip_length)
processed_clip.write_videofile(f"{temp_folder}/04-3.mp4", codec="libx264", fps=video_fps)

foreground_clip.close()
processed_clip.close()

foreground_clip = VideoFileClip(f"{temp_folder}/04-2.mp4")
background_clip =  VideoFileClip(f"{temp_folder}/04-3.mp4")
transition_span = 0.3
left_offest = video_dest_width // 4
processed_clip = VideoClip(appearance_effect, duration=clip_length)

blur_duration = get_transition_span( idx = 4)
clip_duration = clip_length
blurred_clip = processed_clip.fl(apply_increasing_blur)
# Write the processed video to a file
blurred_clip.subclip( 0, clip_length).write_videofile( f"{temp_folder}/04.mp4", codec="libx264", fps=video_fps)

foreground_clip.close()
background_clip.close()
processed_clip.close()
blurred_clip.close()

# ===================== 5. Zoomming In Video ==================================
# Time: 11-15s

clip_length = get_video_length( idx=5)
clip_start, clip_end = get_video_timespan( idx=5)

image = Image.open(os.path.join(folder_path, "5.png")).convert("RGB")

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

dest_width, dest_height = video_dest_width,  video_dest_height
transition_span = clip_length
zoom_dest = 0.8
# Create the video clip using the modified zoom_in_frame function
background_clip = VideoClip(lambda t: zoom_frame(t), duration=clip_length)
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip( clip_start, clip_end).resize((video_dest_width*0.6, video_dest_height*0.6))

foreground_width, foreground_height = foreground_clip.size[0], foreground_clip.size[1]
foreground_x, foreground_y = video_dest_width - foreground_width, video_dest_height - foreground_height

processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t))

blur_duration = get_transition_span( idx = 4)
blurred_start_clip = processed_clip.fl(apply_decreasing_blur)

blur_duration = get_transition_span( idx = 5)
clip_duration = clip_length
blurred_clip = blurred_start_clip.fl(apply_increasing_blur)
# Write the processed video to a file
blurred_clip.subclip( 0, transition_span).write_videofile( f"{temp_folder}/05.mp4", codec="libx264", fps=video_fps)


image.close()
zoom_image.close()
processed_clip.close()
blurred_start_clip.close()
blurred_clip.close()

# ===================== 6. Zoomming In Video ==================================
# Time: 15-18s

clip_length = get_video_length( idx=6)
clip_start, clip_end = get_video_timespan( idx=6)

image = Image.open(os.path.join(folder_path, "6.png")).convert("RGB")

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

dest_width, dest_height = video_dest_width,  video_dest_height
transition_span = clip_length
zoom_dest = 0.8
# Create the video clip using the modified zoom_in_frame function
processed_clip = VideoClip(lambda t: zoom_frame(t), duration=clip_length)

blur_duration = get_transition_span( idx = 5)
blurred_start_clip = processed_clip.fl(apply_decreasing_blur)

blur_duration = get_transition_span( idx = 6)
clip_duration = clip_length
blurred_clip = blurred_start_clip.fl(apply_increasing_blur)

# Write the processed video to a file
blurred_clip.subclip( 0, clip_length).write_videofile( f"{temp_folder}/06.mp4", codec="libx264", fps=video_fps)

image.close()
zoom_image.close()
processed_clip.close()
blurred_start_clip.close()
blurred_clip.close()

# ===================== 7. Zoomming Out Video ==================================
# Time: 18-21

clip_length = get_video_length( idx=7)
clip_start, clip_end = get_video_timespan( idx=7)

image = Image.open(os.path.join(folder_path, "7.png")).convert("RGB")

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

dest_width, dest_height = video_dest_width,  video_dest_height
transition_span = clip_length
zoom_dest = 1.2
# Create the video clip using the modified zoom_in_frame function
processed_clip = VideoClip(lambda t: zoom_frame(t), duration=clip_length)

# processed_clip.write_videofile(f"{temp_folder}/07.mp4", codec="libx264", fps=video_fps)

blur_duration = get_transition_span( idx = 6)
blurred_clip = processed_clip.fl(apply_decreasing_blur)

blurred_clip.crop(0, 0, video_dest_width, video_dest_height//2).write_videofile(f"{temp_folder}/07-1.mp4", codec="libx264", fps=video_fps)
blurred_clip.crop(0, video_dest_height//2, video_dest_width, video_dest_height).write_videofile(f"{temp_folder}/07-2.mp4", codec="libx264", fps=video_fps)

image.close()
zoom_image.close()
processed_clip.close()
blurred_clip.close()

# ===================== 8. Moving Left and Right Videos =======================
# Time: 21-24

clip_length = get_video_length( idx=8)
clip_start, clip_end = get_video_timespan( idx=8)

crop_ratio = 1.3

image = Image.open(os.path.join(folder_path, "8.png")).convert("RGB")
original_width, original_height = image.size

crop_width = original_width
crop_height = int(original_width / crop_ratio)
 
# Calculate left and right coordinates for cropping
top = (original_height - crop_height) // 2
bottom = top + crop_height

# Crop the image
cropped_image = image.crop((0, top, crop_width, bottom))   
src_image = cropped_image.resize( (int(video_dest_height* crop_ratio/2) , video_dest_height//2))

image_width, image_height = src_image.size
zoom_dest = 1
transition_span = clip_length
dest_width, dest_height = video_dest_width, video_dest_height//2
is_to_right = False

upper_video_clip = VideoClip(zoom_and_move_frame, duration=clip_length)
upper_video_clip.write_videofile(f"{temp_folder}/08-1.mp4", codec="libx264", fps=video_fps)
# video_clip.close()

image.close()
src_image.close()


image = Image.open(os.path.join(folder_path, "9.png")).convert("RGB")
original_width, original_height = image.size

crop_width = original_width
crop_height = int(original_width / crop_ratio)
 
# Calculate left and right coordinates for cropping
top = (original_height - crop_height) // 2
bottom = top + crop_height

# Crop the image
cropped_image = image.crop((0, top, crop_width, bottom))   
src_image = cropped_image.resize( (int(video_dest_height* crop_ratio/2) , video_dest_height//2))

image_width, image_height = src_image.size
zoom_dest = 1
transition_span = clip_length
dest_width, dest_height = video_dest_width, video_dest_height//2
is_to_right = True

lower_video_clip = VideoClip(zoom_and_move_frame, duration=clip_length)
lower_video_clip.write_videofile(f"{temp_folder}/08-2.mp4", codec="libx264", fps=video_fps)
# video_clip.close()


image.close()
src_image.close()
upper_video_clip.close()
lower_video_clip.close()

# Concatenate Video 7 & 8

commands = [
    f'ffmpeg -i 07-1.mp4 -i 08-1.mp4 -filter_complex "[0][1]xfade=transition=wipeleft:duration={get_transition_span(idx=7)}:offset={video_spans[6]},format=yuv420p" 07-08-1.mp4',
    f'ffmpeg -i 07-2.mp4 -i 08-2.mp4 -filter_complex "[0][1]xfade=transition=revealright:duration={get_transition_span(idx=7)}:offset={video_spans[6]},format=yuv420p" 07-08-2.mp4',
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
        
upper_video_clip = VideoFileClip(f"{temp_folder}/07-08-1.mp4")
lower_video_clip = VideoFileClip(f"{temp_folder}/07-08-2.mp4")
composed_clip = clips_array([[upper_video_clip], [lower_video_clip]])

blur_duration = get_transition_span( idx = 8)
clip_duration = composed_clip.duration
blurred_clip = composed_clip.fl(apply_increasing_blur)

blurred_clip.write_videofile(f"{temp_folder}/07-08.mp4", codec="libx264", fps=video_fps)

upper_video_clip.close()
lower_video_clip.close()
composed_clip.close()
blurred_clip.close()
# ===================== 9. Zoomming Out Video ==================================
# Time: 24-29
print('==================== 9. Zoomming Out Video START ===============================')
clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4"))
clip_length = get_video_length( idx=9)
clip_start, clip_end = get_video_timespan( idx=9)
print(clip_end)
clip_end = clip.duration
print(clip_end)
avatar_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip( clip_start, clip_end)

blur_duration = get_transition_span( idx = 8)
blurred_start_clip = avatar_clip.fl(apply_decreasing_blur)

blur_duration = get_transition_span( idx = 9)
clip_duration = avatar_clip.duration
blurred_clip = blurred_start_clip.fl(apply_increasing_blur)

blurred_clip.write_videofile(f"{temp_folder}/09.mp4", codec="libx264", fps=video_fps)
print('==================== 9. Zoomming Out Video END ===============================')
avatar_clip.close()
blurred_start_clip.close()
blurred_clip.close()

# ================ 10. Blur video ===============================
# Time 29-35s

clip_length = get_video_length( idx=10)
clip_start, clip_end = get_video_timespan( idx=10)

action_delay = 1

def blur_frame(frame):
    return cv2.GaussianBlur(frame, (51, 51), 10)

blur_amount = 3

clip = VideoFileClip(os.path.join(folder_path, "ss.MP4")).subclip(8, 8+clip_length)
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

processed_clip =  concatenate_videoclips([ cropped_clip.subclip( 0, background_clip.duration - foreground_clip.duration)
                                          , background_clip.subclip( background_clip.duration - foreground_clip.duration, background_clip.duration).fl(lambda gf, t: add_foreground(gf(t), t))])

blur_duration = get_transition_span( idx = 9)
blurred_clip = processed_clip.fl(apply_decreasing_blur)
# Write the processed video to a file
blurred_clip.subclip(0, clip_length).write_videofile( f"{temp_folder}/10.mp4", codec="libx264", fps=video_fps)

background_clip.close()
foreground_clip.close()
processed_clip.close()
blurred_clip.close()

# =============================================================================
# ====================  Concantenate  Background Videos  ======================
# =============================================================================

video_clip_names = ['01-02.mp4', '03.mp4', '04.mp4', '05.mp4', '06.mp4', '07-08.mp4', '09.mp4', '10.mp4']

video_clip_names = [os.path.join(temp_folder, video_clip_name) for video_clip_name in video_clip_names]

background_video = os.path.join(temp_folder, "background_video.mp4")
command = ['xvfb-run -s "-screen 0 1024x768x24" ', 'ffmpeg-concat', '-T', "./templates/template3/input/transition.json",
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

# =============================================================================
# ======================  Add Foreground Video(Avatar)  =======================
# =============================================================================


background_clip = VideoFileClip(f"{temp_folder}/background_video.mp4")
processed_clips = []
    
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip( 15, 21).resize((video_dest_width*0.75, video_dest_height*0.75))

foreground_width, foreground_height = foreground_clip.size[0], foreground_clip.size[1]
foreground_x, foreground_y = ( video_dest_width - foreground_width)//2, video_dest_height - foreground_height

processed_clip = background_clip.subclip(15, 21).fl(lambda gf, t: add_foreground(gf(t), t))
final_clip = concatenate_videoclips([background_clip.subclip(0, 15), processed_clip, background_clip.subclip(21, background_clip.duration)])
final_clip.write_videofile( f"{temp_folder}/overlapped_video.mp4", codec="libx264", fps=video_fps)

for file_name in os.listdir(temp_folder):
    file_path = os.path.join(temp_folder, file_name)

    if file_name != "overlapped_video.mp4":
        os.remove(file_path)