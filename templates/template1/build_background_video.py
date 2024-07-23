# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 00:26:48 2024

@author: codemaven
"""

import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoClip, VideoFileClip
import subprocess
import json
import os
import argparse

parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
args = parser.parse_args()

folder_path = args.folderPath
# add folder path to temp_folder

# Load your image using PIL
video_fps = 30
video_dest_width = 2180
video_dest_height = 3840
wipe_left_time = 400

with open('./templates/template1/input/transition_span.json', 'r') as f:
    transition_list = json.load(f)
intervals = [3, 2, 2, 3, 3, 1, 2.5, 2.5, 2, 3.5, 4.5, 6]
temp_folder = os.path.join(folder_path, "temp")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)

def get_durations(idx):
    if idx == 1:
        return intervals[0] + transition_list[0].get("duration") / 2000
    elif idx == len(intervals):
        return intervals[-1] + transition_list[-1].get("duration") / 2000
    elif idx in ( 8, 9, 11):
        return intervals[idx - 1] + (transition_list[idx - 2].get(
            "duration") + wipe_left_time) / 2000
    elif idx > 1 and idx < len(intervals):
        return intervals[idx - 1] + (transition_list[idx - 2].get(
            "duration") + transition_list[idx - 1].get("duration")) / 2000
    return 0


def get_timestamp_with_transtion_span(idx):
    if idx == 1:
        return 0, intervals[0] + transition_list[0].get("duration") / 2000
    elif idx == len(intervals):
        return sum(
            intervals[:-1]) - transition_list[-1].get("duration") / 2000, sum(intervals)
    elif idx > 1 and idx < len(intervals):
        return sum(intervals[:idx - 1]) - transition_list[idx - 2].get("duration") / \
            2000, sum(intervals[:idx]) + transition_list[idx - 1].get("duration") / 2000
    return 0, 0


def get_timestamp_without_transtion_span(idx):
    if idx == 1:
        return 0, intervals[0] - transition_list[0].get("duration") / 2000
    elif idx == len(intervals):
        return transition_list[-1].get("duration") / 1000, get_durations(idx)
    elif idx > 1 and idx < len(intervals):
        return transition_list[idx - 2].get("duration") / 1000, get_durations(
            idx) - transition_list[idx - 1].get("duration") / 1000
    return 0, 0


def write_videoclips(video_clip, idx):
    start, end = get_timestamp_without_transtion_span(idx)
    # if ids is 11 or 12 add temp audio folder to path
    if idx in (11, 12):
        video_clip.subclip(
            0, video_clip.duration).write_videofile(
                f"{temp_folder}/{idx:02d}.mp4", fps=video_fps, temp_audiofile=f"{temp_folder}/{idx}.mp3", remove_temp=True)
    else:
        video_clip.subclip(
            0, video_clip.duration).write_videofile(
                f"{temp_folder}/{idx:02d}.mp4", fps=video_fps)
                
# Create the video clip for moving right effect
def moving_right_frame(t):
    new_left = int((image_width - video_dest_width) * t /
                   duration)  # Moves from right to left
    cropped_image = image.crop(
        (image_width -
         new_left -
         video_dest_width,
         0,
         image_width -
         new_left,
         video_dest_height))
    return np.array(cropped_image)

def move_from_top(t):
    if t >= duration:
        t = duration
    new_height = int(video_dest_height * t // (2 * duration))
    cropped_image = image.crop(
        (0,
         video_dest_height //
         2 -
         new_height,
         video_dest_width,
         video_dest_height //
         2))
    back_image.paste(cropped_image, (0, 0))
    return np.array(back_image)


def drop_from_top(t):
    if t >= duration:
        t = duration
    new_height = int(video_dest_height * t // (2 * duration))
    cropped_image = image.crop(
        (0,
         video_dest_height //
         2 -
         new_height,
         video_dest_width,
         video_dest_height //
         2))
    back_image.paste(cropped_image, (0, 0))
    return np.array(back_image)


def pop_from_bottom(t):
    if t >= duration:
        t = duration
    new_height = int(video_dest_height * t // (2 * duration))
    cropped_image = image.crop((0, 0, video_dest_width, new_height))
    back_image.paste(cropped_image, (0, video_dest_height - new_height))
    return np.array(back_image)


# ================== 1. Image to moving right video(1) ========================
# Time : 0-3s
image = Image.open(os.path.join(folder_path, "1.png"))
image_width, image_height = image.size
duration = get_durations(idx=1)

video_clip = VideoClip(moving_right_frame, duration=duration)
write_videoclips(video_clip, idx=1)
video_clip.close()

# ================== 2. Image to moving right video(2) ========================
# Time : 3-5s
image = Image.open(os.path.join(folder_path, "2.png"))
image_width, image_height = image.size
duration = get_durations(idx=2)

video_clip = VideoClip(moving_right_frame, duration=duration)
write_videoclips(video_clip, idx=2)
video_clip.close()

# ================== 3. Resize Video into destination size ====================
# Time : 5-7s
clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).subclip(0, get_durations(idx=3))
aspect_ratio = clip.w / clip.h
dest_ratio = video_dest_width / video_dest_height

if aspect_ratio > dest_ratio:
    new_width = int(clip.h * dest_ratio)
    new_height = clip.h
else:
    new_width = clip.w
    new_height = int(clip.w / dest_ratio)

cropped_clip = clip.crop(
    x_center=(
        clip.w // 2),
    y_center=(
        clip.h // 2),
    width=new_width,
    height=new_height).resize(
    (video_dest_width,
     video_dest_height))
write_videoclips(cropped_clip, idx=3)
clip.close()


# ================== 4. Image to moving right video(3) ========================
# Time : 7-10s
image = Image.open(os.path.join(folder_path, "3.png"))
image_width, image_height = image.size
duration = get_durations(idx=4)

video_clip = VideoClip(moving_right_frame, duration=duration)
write_videoclips(video_clip, idx=4)
video_clip.close()

# ================== 5. Image to zooming in video =============================
# Time : 10-13s
def zoom_out_frame(t):
    # Use a smoother easing function for the zoom factor
    zoom_factor = 1 - ((1 - zoom_dest) * (t / duration)**0.5)
    
    # Calculate new dimensions using floating-point arithmetic
    new_width = image_width * zoom_factor
    new_height = image_height * zoom_factor
    
    # Center the cropped area with floating-point precision
    left = (image_width - new_width) / 2
    top = (image_height - new_height) / 2
    right = left + new_width
    bottom = top + new_height
    
    # Crop the image to the new dimensions
    cropped_image = image.crop((left, top, right, bottom))
    
    # Calculate aspect ratio of the video destination
    video_aspect_ratio = video_dest_width / video_dest_height
    image_aspect_ratio = new_width / new_height

    # Resize with aspect ratio correction
    if image_aspect_ratio > video_aspect_ratio:
        # Image is wider than the video aspect ratio
        resize_height = video_dest_height
        resize_width = int(new_width * (video_dest_height / new_height))
    else:
        # Image is taller than the video aspect ratio
        resize_width = video_dest_width
        resize_height = int(new_height * (video_dest_width / new_width))
    
    resized_image = cropped_image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
    
    # Center crop the resized image to the exact video destination size
    left = (resize_width - video_dest_width) / 2
    top = (resize_height - video_dest_height) / 2
    right = left + video_dest_width
    bottom = top + video_dest_height

    final_image = resized_image.crop((left, top, right, bottom))
    
    return np.array(final_image)

# Load the image and convert it to RGB
rgba_image = Image.open(os.path.join(folder_path, "4.png"))
image = rgba_image.convert("RGB")
image_width, image_height = image.size
duration = get_durations(idx=5)
zoom_dest = 0.7

# Create the video clip using the modified zoom_out_frame function
video_clip = VideoClip(lambda t: zoom_out_frame(t), duration=duration)

# Write the video clip to a file (assuming the write_videoclips function is defined)
write_videoclips(video_clip, idx=5)

# Close the video clip to free resources
video_clip.close()


# ================== 6. Image to dropping down from top video =================
# Time : 13-14s

back_image = Image.fromarray(video_clip.get_frame(video_clip.duration - 0.1))

fore_image = Image.open(os.path.join(folder_path, "5.png"))
image_width, image_height = fore_image.size
duration = 1 / 4

crop_height = int(image_width * video_dest_height / (video_dest_width * 2))
image = fore_image.crop(
    (0,
     image_height -
     crop_height,
     image_width,
     image_height)).resize(
    (video_dest_width,
     video_dest_height //
     2))

video_clip = VideoClip(drop_from_top, duration=get_durations(idx=6))
write_videoclips(video_clip, idx=6)


# ================== 7. Image to poping up from bottom video ==================
# Time : 14-16.5s
back_image = Image.fromarray(video_clip.get_frame(0.99))
fore_image = Image.open(os.path.join(folder_path, "6.png")).convert("RGB")
image_width, image_height = fore_image.size
duration = 1 / 4

crop_height = int(image_width * video_dest_height / (video_dest_width * 2))
image = fore_image.crop(
    (0, 0, image_width, crop_height)).resize(
        (video_dest_width, video_dest_height // 2))

video_clip = VideoClip(pop_from_bottom, duration=get_durations(idx=7))
write_videoclips(video_clip, idx=7)
video_clip.close()

# ================== 8. Video subclip from avatar =============================
# Time : 16.5-19s


def split_top_bottom(t):
    duration = end_time - start_time
    if t <= start_time:
        return np.array(front_image)
    elif t >= end_time:
        return np.array(Image.fromarray(video_clip.get_frame(t)))
    else:
        t -= start_time
    new_height = int(video_dest_height * (duration - t) // (2 * duration))
    top_image = front_image.crop(
        (0,
         video_dest_height //
         2 -
         new_height,
         video_dest_width,
         video_dest_height //
         2))
    bottom_image = front_image.crop(
        (0,
         video_dest_height //
         2,
         video_dest_width,
         video_dest_height //
         2 +
         new_height))
    back_image = Image.fromarray(video_clip.get_frame(t))
    back_image.paste(top_image, (0, 0))
    back_image.paste(bottom_image, (0, video_dest_height - new_height))

    image = np.array(back_image)
    # Define the region to keep sharp (coordinates of the bounding box)
    region_box = (
        0,
        new_height,
        video_dest_width,
        video_dest_height -
        new_height)

    # Define the strength of blur transitions
    # max_blur_strength = 80
    # blur_step = 5
    max_blur_strength = 0
    blur_step = 1
    # Gradually decrease the intensity of the blur as you move away from the
    # border of the region box
    blurred_image = image.copy()
    for i in range(0, max_blur_strength, blur_step):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (region_box[0], region_box[1] +
                             (max_blur_strength -
                              i)), (region_box[2], region_box[3] -
                                    (max_blur_strength -
                                     i)), 255, -
                      1)

        blurred_image_copy = blurred_image.copy()
        blurred_image_copy = cv2.GaussianBlur(blurred_image_copy, (int(
            20 * t) * 2 + 1, int(20 * t) * 2 + 1), i)  # Adjust kernel size and sigma as needed
        blurred_image = np.where(
            mask[..., None] != 0, blurred_image, blurred_image_copy)

    return blurred_image
    # return np.array( result_image)

start_time = 0.2
end_time = 0.8
front_image = Image.fromarray(video_clip.get_frame(2.49))
video_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).subclip(
    get_timestamp_with_transtion_span(8)[0],
    get_timestamp_with_transtion_span(8)[1]).resize(
        (video_dest_width,
         video_dest_height))

result_clip = VideoClip(split_top_bottom, duration=get_durations(idx=8))

write_videoclips(result_clip, idx=8)
result_clip.close()

# ================== 9. Image to zooming out and moving right video ===========
# Time : 19-21s
def moving_and_zoom_frame(t):
    new_left = int(image_width / 4) - int((image_width - \
                   video_dest_width) * t / duration) # Moves from right to left

    zoom_factor = 1 - (1 - 1 / zoom_dest) * (duration - t) / \
        duration  # decrease zoom gradually over time

    new_width = int(video_dest_width * zoom_factor)
    new_height = int(video_dest_height * zoom_factor)

    left = (image_width - new_width) // 2 + new_left
    top = (image_height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((video_dest_width, video_dest_height))
    return np.array(resized_image)

rgba_image = Image.open(os.path.join(folder_path, "7.png"))
zoom_dest = 1.3
duration = get_durations(idx=9)

image = rgba_image.convert("RGB")
image_width, image_height = image.size
video_clip.close()

video_clip = VideoClip(moving_and_zoom_frame, duration=duration)
write_videoclips(video_clip, idx=9)
video_clip.close()

# ================== 10. Resize Video into destination size ==============
# Time : 21-24.5s

clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).subclip(0, get_durations(idx=10))
aspect_ratio = clip.w / clip.h
dest_ratio = video_dest_width / video_dest_height

if aspect_ratio > dest_ratio:
    new_width = int(clip.h * dest_ratio)
    new_height = clip.h
else:
    new_width = clip.w
    new_height = int(clip.w / dest_ratio)

cropped_clip = clip.crop(
    x_center=(
        clip.w // 2),
    y_center=(
        clip.h // 2),
    width=new_width,
    height=new_height).resize(
    (video_dest_width,
     video_dest_height))
write_videoclips(cropped_clip, idx=10)
clip.close()


# ================== 11. Video subclip from avatar =======================
# Time : 24.5-29s
clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4"))

start_time, end_time = get_timestamp_with_transtion_span(11)

end_time = clip.duration

clip = clip.subclip(start_time, end_time)

offset_for_avatar = clip.duration


aspect_ratio = clip.w / clip.h
dest_ratio = video_dest_width / video_dest_height

if aspect_ratio > dest_ratio:
    new_width = int(clip.size[1] * dest_ratio)
    new_height = clip.size[1]
else:
    new_width = clip.size[0]
    new_height = int(clip.size[0] / dest_ratio)

cropped_clip = clip.crop(
    x_center=(
        clip.w // 2),
    y_center=(
        clip.h // 2),
    width=new_width,
    height=new_height).resize(
    (video_dest_width,
     video_dest_height))
write_videoclips(cropped_clip, idx=11)
clip.close()
# ================== 12. Blur video ===========================================
# Time : 29-35s

def blur_frame(frame):
    return cv2.GaussianBlur(frame, (51, 51), 10)

blur_amount = 3
clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).subclip(0, get_durations(idx=12))
aspect_ratio = clip.w / clip.h
dest_ratio = video_dest_width / video_dest_height

if aspect_ratio > dest_ratio:
    new_width = int(clip.size[1] * dest_ratio)
    new_height = clip.size[1]
else:
    new_width = clip.size[0]
    new_height = int(clip.size[0] / dest_ratio)

cropped_clip = clip.crop(
    x_center=(
        clip.w // 2),
    y_center=(
        clip.h // 2),
    width=new_width,
    height=new_height).resize(
    (video_dest_width,
     video_dest_height))

background_clip = cropped_clip.fl_image(blur_frame)
clip.close()
# set parameter for replace_green_background function


def replace_green_background(frame, time):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Define lower and upper bounds for green color in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask_inverse = cv2.bitwise_not(mask)
    foreground_frame = cv2.bitwise_and(frame, frame, mask=mask_inverse)

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
        background_clip.get_frame(time),
        background_clip.get_frame(time),
        mask=total_mask_inverse)
    final_frame = cv2.bitwise_or(foreground_with_mask, background_with_mask)
    return final_frame


foreground_clip = VideoFileClip(
    os.path.join(folder_path, "action.mp4")).subclip(0, get_durations(idx=12))

foreground_width = background_clip.size[0]
foreground_height = int(
    foreground_clip.size[1] *
    foreground_width /
    foreground_clip.size[0])
foreground_clip = foreground_clip.resize((foreground_width, foreground_height))

foreground_x = 0
foreground_y = (background_clip.size[1] - foreground_height) // 2
# Apply the replace_green_background function to each frame of the
# foreground video
processed_clip = foreground_clip.fl(
    lambda gf, t: replace_green_background(gf(t), t))
write_videoclips(processed_clip, idx=12)
foreground_clip.close()

avatar_clip = VideoFileClip(os.path.join(temp_folder, "11.mp4"))

# List of FFmpeg commands
commands = [
    f'ffmpeg -i 08.mp4 -i 09.mp4 -filter_complex "[0][1]xfade=transition=smoothleft:duration={wipe_left_time/1000}:offset=2.5,format=yuv420p" 08-09.mp4',
    f'ffmpeg -i 11.mp4 -i 12.mp4 -filter_complex "[0][1]xfade=transition=smoothleft:duration={wipe_left_time/1000}:offset={avatar_clip.duration - 0.4},format=yuv420p" 11-12.mp4',
    f'ffmpeg -i 08-09.mp4 -i 10.mp4 -filter_complex "[0][1]xfade=transition=smoothleft:duration={wipe_left_time/1000}:offset=4.5,format=yuv420p" 08-09-10.mp4',
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

# ====================== ADD TRANSITION BETWEEN THEM===========================
video_clip_names = [f"{i+1:02d}.mp4" for i in range(7)]
video_clip_names.append("08-09-10.mp4")
video_clip_names.append("11-12.mp4")

#add temp_folder to the video_clip_names
video_clip_names = [os.path.join(temp_folder, video_clip_name) for video_clip_name in video_clip_names]

background_video = os.path.join(temp_folder, "background_video.mp4")

command = ['ffmpeg-concat', '-T', "./templates/template1/input/transition.json",
           '-o', background_video] + video_clip_names
print(" ".join(command))
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
        print(f"command includes errors:  {completed_process}")
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")

for file_name in os.listdir(temp_folder):
    file_path = os.path.join(temp_folder, file_name)

    if file_name != "background_video.mp4":
        os.remove(file_path)
