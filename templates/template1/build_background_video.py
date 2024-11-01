# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 00:26:48 2024

@author: codemaven
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from moviepy.editor import VideoClip, VideoFileClip
import subprocess
import json
import os
import argparse

parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
args = parser.parse_args()


# Aspect ratios for comparison
ASPECT_RATIOS = {
    "Square": (1, 1),
    "Tall": (9, 16),
    "Wide": (4, 3),
    "Half": (9, 8)
}

folder_path = args.folderPath
# add folder path to temp_folder

# Load your image using PIL
video_fps = 30
video_dest_width = 1216
video_dest_height = 2160
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


def get_closest_aspect_ratio(image):
    """
    Returns the closest predefined aspect ratio to the given image dimensions.
    
    Parameters:
    - image (PIL.Image): The input image.
    
    Returns:
    - str: The name of the closest aspect ratio ('Square', 'Tall', or 'Wide').
    - float: scale.
    """
    # Get image dimensions
    image_width, image_height = image.size

    # Calculate the aspect ratio of the image
    image_aspect_ratio = image_width / image_height

    # Find the closest match by calculating the difference with each target aspect ratio
    closest_ratio_name, closest_ratio = min(
        ASPECT_RATIOS.items(),
        key=lambda ratio: abs(image_aspect_ratio - ratio[1][0] / ratio[1][1])
    )

    return closest_ratio_name, video_dest_height * image_width / ( video_dest_width * image_height)

def resize_with_scaled_target(input_source, target="Tall", fill_blank=False, scale=1.0, scale_direction="width"):
    """
    Adjusts the given image or video clip to match a target aspect ratio ('Square', 'Tall', or 'Wide') 
    with optional scaling applied to either width or height only.
    Crops to center or adds black margins depending on the fill_blank parameter.

    Parameters:
    - input_source (PIL.Image or VideoFileClip): The input image or video clip.
    - target (str): The target aspect ratio, one of 'Square', 'Tall', or 'Wide'.
    - fill_blank (bool): If True, adds black margins to fill the aspect ratio;
                         if False, crops to center.
    - scale (float): Scaling factor for the target aspect ratio (e.g., 1.2 for 20% wider/taller).
    - scale_direction (str): Specifies which dimension to scale; choose 'width' or 'height'.

    Returns:
    - PIL.Image or VideoFileClip: The adjusted image or video clip.
    """
    
    def _resize_image(image, target_aspect_ratio, fill_blank):
        """
        Helper function to resize an image to the target aspect ratio.
        """
        image_width, image_height = image.size
        image_aspect_ratio = image_width / image_height

        if image_aspect_ratio > target_aspect_ratio:
            # Image is wider than target; adjust width to match aspect ratio
            new_height = image_height
            new_width = int(new_height * target_aspect_ratio)
        else:
            # Image is taller than target; adjust height to match aspect ratio
            new_width = image_width
            new_height = int(new_width / target_aspect_ratio)
        
        if fill_blank:
            # Add black margins to match target aspect ratio if needed
            adjusted_image = ImageOps.pad(image, (new_width, new_height), color=(0, 0, 0), centering=(0.5, 0.5))
        else:
            # Crop the image to the largest possible centered area with the target aspect ratio
            adjusted_image = ImageOps.fit(image, (new_width, new_height), centering=(0.5, 0.5))
        
        return adjusted_image

    def _resize_video(video_clip, target_aspect_ratio, fill_blank):
        """
        Helper function to resize a video to the target aspect ratio by applying a function to each frame.
        """
        def process_frame(frame):
            # Convert the frame to a PIL image for processing
            image = Image.fromarray(frame)
            adjusted_image = _resize_image(image, target_aspect_ratio, fill_blank)
            return np.array(adjusted_image)
        
        # Apply the resize function to each frame of the video
        adjusted_video = video_clip.fl_image(process_frame)
        return adjusted_video
    
    # Validate target aspect ratio and scale_direction
    if target not in ASPECT_RATIOS:
        raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")
    if scale_direction not in {"width", "height"}:
        raise ValueError("Invalid scale direction. Choose 'width' or 'height'.")
    
    # Get the base target aspect ratio and apply scaling to one dimension
    base_width_ratio, base_height_ratio = ASPECT_RATIOS[target]
    if scale_direction == "width":
        scaled_width_ratio = base_width_ratio * scale
        scaled_height_ratio = base_height_ratio
    else:  # scale_direction == "height"
        scaled_width_ratio = base_width_ratio
        scaled_height_ratio = base_height_ratio * scale
    target_aspect_ratio = scaled_width_ratio / scaled_height_ratio
    
    # Check if input is an image
    if isinstance(input_source, Image.Image):
        return _resize_image(input_source, target_aspect_ratio, fill_blank)
    elif isinstance(input_source, VideoFileClip):
        return _resize_video(input_source, target_aspect_ratio, fill_blank)
    else:
        raise TypeError("Unsupported input type. Provide either an image (PIL.Image) or video clip (VideoFileClip).")

def sliding_frame(t, direction, image, video_dest_width, video_dest_height, duration, scale=1.0, scale_direction="width"):
    """
    Returns a frame that slides the image in the specified direction, resizing it to match
    the video destination dimensions without preserving aspect ratio.

    Parameters:
    - t (float): Current time in the video.
    - direction (str): Direction of slide ('right', 'left', 'up', 'down').
    - image (PIL.Image): The image to be resized and slid.
    - video_dest_width (int): Width of the video frame.
    - video_dest_height (int): Height of the video frame.
    - duration (float): Duration of the sliding effect.

    Returns:
    - np.array: The resized and cropped image frame for the current time t.
    """
    if scale_direction == "width":
        resize_width = int( video_dest_width * scale)
        resize_height = video_dest_height
    else:  # scale_direction == "height"
        resize_width = video_dest_width
        resize_height = int( video_dest_height * scale)
        
    resized_image = image.resize((resize_width, resize_height))

    # Get dimensions of the resized image
    image_width, image_height = resized_image.size
    
    # Calculate sliding offsets based on the direction
    if direction == "right":
        # Slide from left to right
        new_left = int((image_width - video_dest_width) * t / duration)
        cropped_image = resized_image.crop((new_left, 0, new_left + video_dest_width, video_dest_height))
        
    elif direction == "left":
        # Slide from right to left
        new_left = int((image_width - video_dest_width) * (1 - t / duration))
        cropped_image = resized_image.crop((new_left, 0, new_left + video_dest_width, video_dest_height))
        
    elif direction == "up":
        # Slide from bottom to top
        new_top = int((image_height - video_dest_height) * (1 - t / duration))
        cropped_image = resized_image.crop((0, new_top, video_dest_width, new_top + video_dest_height))
        
    elif direction == "down":
        # Slide from top to bottom
        new_top = int((image_height - video_dest_height) * t / duration)
        cropped_image = resized_image.crop((0, new_top, video_dest_width, new_top + video_dest_height))
        
    else:
        raise ValueError("Invalid direction. Choose 'right', 'left', 'up', or 'down'.")
    
    return np.array(cropped_image)

def zoom_frame(t, image, video_dest_width, video_dest_height, duration, zoom_dest=1.0):
    """
    Returns a frame that applies a zoom effect (in or out) on the image, centered for smooth transitions.

    Parameters:
    - t (float): Current time in the video.
    - image (PIL.Image): The image to be zoomed.
    - video_dest_width (int): Width of the video frame.
    - video_dest_height (int): Height of the video frame.
    - duration (float): Duration of the zoom effect.
    - zoom_dest (float): Target zoom factor. 
                         >1 means zoom-in to original size; <1 means zoom-out from original size.

    Returns:
    - np.array: The resized image frame for the current time t.
    """
    image_width, image_height = image.size

    # Calculate zoom factor based on whether zooming in or out
    if zoom_dest > 1:  # Zooming in to the original size
        zoom_factor = 1 / (zoom_dest - ((zoom_dest - 1) * (t / duration)**0.5))
    elif zoom_dest < 1:  # Zooming out from the original size
        zoom_factor = 1 - ((1 - zoom_dest) * (t / duration)**0.5)
    else:
        zoom_factor = 1  # No zoom effect if zoom_dest is 1

    # Calculate new dimensions for the zoomed area
    new_width = image_width * zoom_factor
    new_height = image_height * zoom_factor

    # Center-calculate cropping coordinates
    left = (image_width - new_width) / 2
    top = (image_height - new_height) / 2
    right = left + new_width
    bottom = top + new_height

    # Crop and resize the image to fit video destination dimensions
    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((video_dest_width, video_dest_height), Image.Resampling.LANCZOS)
    
    return np.array(resized_image)


# ================== 1. Image to moving right video(1) ========================

# Time : 0-3s
image = Image.open(os.path.join(folder_path, "1.png")).convert("RGB")

closest_ratio_name, scale = get_closest_aspect_ratio( image)
duration = get_durations(idx=1)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    video_clip = VideoClip(
        lambda t: zoom_frame(
            t,
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            zoom_dest=0.7  # Target zoom level for final frame (e.g., <1 for zoom out)
        ),
        duration=duration
    )
else:
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    # adjusted_image.show()
    video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            scale = scale,
            scale_direction = 'width'
        ),
        duration=duration
    )
    
write_videoclips(video_clip, idx=1)
video_clip.close()

# ================== 2. Image to moving right video(2) ========================
# Time : 3-5s
image = Image.open(os.path.join(folder_path,"2.png")).convert("RGB")
closest_ratio_name, scale = get_closest_aspect_ratio( image)
duration = get_durations(idx=2)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    
    video_clip = VideoClip(
        lambda t: zoom_frame(
            t,
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            zoom_dest=1.3  # Target zoom level for final frame (e.g., <1 for zoom out)
        ),
        duration=duration
    )
else:
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="left",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            scale = scale,
            scale_direction = 'width'
        ),
        duration=duration
    )
write_videoclips(video_clip, idx=2)
video_clip.close()

# ================== 3. Resize Video into destination size ====================

# Time : 5-7s
clip = VideoFileClip(os.path.join(folder_path,"ss.mp4")).subclip(0, get_durations(idx=3))
adjusted_video = resize_with_scaled_target(clip).resize((video_dest_width, video_dest_height))
write_videoclips( adjusted_video, idx=3)
clip.close()


# ================== 4. Image to moving right video(3) ========================
# Time : 7-10s
image = Image.open(os.path.join(folder_path,"3.png")).convert("RGB")
closest_ratio_name, scale = get_closest_aspect_ratio( image)
duration = get_durations(idx=4)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    
    video_clip = VideoClip(
        lambda t: zoom_frame(
            t,
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            zoom_dest=1.3  # Target zoom level for final frame (e.g., <1 for zoom out)
        ),
        duration=duration
    )
else:
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="left",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            scale = scale,
            scale_direction = 'width'
        ),
        duration=duration
    )
    
write_videoclips(video_clip, idx=4)
video_clip.close()

# ================== 5. Image to zooming in video =============================
# Time : 10-13s

# Load the image and convert it to RGB
image = Image.open(os.path.join(folder_path,"4.png")).convert("RGB")
closest_ratio_name, scale = get_closest_aspect_ratio( image)
duration = get_durations(idx=5)

# Create the video clip using the modified zoom_out_frame function
if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    video_clip = VideoClip(
        lambda t: zoom_frame(
            t,
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            zoom_dest=0.7  # Target zoom level for final frame (e.g., <1 for zoom out)
        ),
        duration=duration
    )
else:
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            scale = scale,
            scale_direction = 'width'
        ),
        duration=duration
    )
    
write_videoclips(video_clip, idx=5)
video_clip.close()

# ================== 6. Image to dropping down from top video =================
# Time : 13-14s

back_image = Image.fromarray(video_clip.get_frame(video_clip.duration - 0.1))
fore_image = Image.open(os.path.join(folder_path,"5.png")).convert("RGB")

image = resize_with_scaled_target( fore_image, target="Half").resize((video_dest_width, video_dest_height //2))

duration = 1 / 4
video_clip = VideoClip(drop_from_top, duration=get_durations(idx=6))
write_videoclips(video_clip, idx=6)


# ================== 7. Image to poping up from bottom video ==================
# Time : 14-16.5s
back_image = Image.fromarray(video_clip.get_frame(0.99))
fore_image = Image.open(os.path.join(folder_path,"6.png")).convert("RGB")

image = resize_with_scaled_target( fore_image, target="Half").resize((video_dest_width, video_dest_height //2))
duration = 1 / 4

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
video_clip = VideoFileClip(os.path.join(folder_path,"bg_avatar.mp4")).subclip(
    get_timestamp_with_transtion_span(8)[0],
    get_timestamp_with_transtion_span(8)[1]).resize(
        (video_dest_width,
         video_dest_height))

result_clip = VideoClip(split_top_bottom, duration=get_durations(idx=8))

write_videoclips(result_clip, idx=8)
result_clip.close()

# ================== 9. Image to zooming out and moving right video ===========
# Time : 19-21s

image = Image.open(os.path.join(folder_path,"7.png")).convert("RGB")
closest_ratio_name, scale = get_closest_aspect_ratio( image)
duration = get_durations(idx=9)

# Create the video clip using the modified zoom_out_frame function
if closest_ratio_name == "Tall":
    closest_ratio_name = resize_with_scaled_target(image, target=closest_ratio_name)
    video_clip = VideoClip(
        lambda t: zoom_frame(
            t,
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            zoom_dest=1.3  # Target zoom level for final frame (e.g., <1 for zoom out)
        ),
        duration=duration
    )
else:
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=duration,
            scale = scale,
            scale_direction = 'width'
        ),
        duration=duration
    )
    
write_videoclips(video_clip, idx=9)
video_clip.close()

# def moving_and_zoom_frame(t):
#     new_left = int(image_width / 4) - int((image_width - \
#                    video_dest_width) * t / duration) # Moves from right to left

#     zoom_factor = 1 - (1 - 1 / zoom_dest) * (duration - t) / \
#         duration  # decrease zoom gradually over time

#     new_width = int(video_dest_width * zoom_factor)
#     new_height = int(video_dest_height * zoom_factor)

#     left = (image_width - new_width) // 2 + new_left
#     top = (image_height - new_height) // 2
#     right = left + new_width
#     bottom = top + new_height

#     cropped_image = image.crop((left, top, right, bottom))
#     resized_image = cropped_image.resize((video_dest_width, video_dest_height))
#     return np.array(resized_image)

# rgba_image = Image.open(os.path.join(folder_path,"9.png").resize((1500, 1500))
# zoom_dest = 1.3
# duration = get_durations(idx=9)

# image = rgba_image.convert("RGB")
# image_width, image_height = image.size
# video_clip.close()


# ================== 10. Resize Video into destination size ==============
# Time : 21-24.5s

clip = VideoFileClip(os.path.join(folder_path,"ss.mp4")).subclip(0, get_durations(idx=10))
adjusted_video = resize_with_scaled_target(clip).resize((video_dest_width, video_dest_height))
write_videoclips(adjusted_video, idx=10)
clip.close()

# ================== 11. Video subclip from avatar =======================
# Time : 24.5-29s
clip_start = get_timestamp_with_transtion_span(11)[0]

clip = VideoFileClip(os.path.join(folder_path,"bg_avatar.mp4"))

clip_end = clip.duration

clip = clip.subclip(clip_start, clip_end)

adjusted_video = resize_with_scaled_target(clip).resize((video_dest_width, video_dest_height))
write_videoclips(adjusted_video, idx=11)
clip.close()
# ================== 12. Blur video ===========================================
# Time : 29-35s

def blur_frame(frame):
    return cv2.GaussianBlur(frame, (51, 51), 10)

blur_amount = 3
clip = VideoFileClip(os.path.join(folder_path,"ss.mp4")).subclip(0, get_durations(idx=12))
adjusted_video = resize_with_scaled_target(clip).resize((video_dest_width, video_dest_height))
background_clip = adjusted_video.fl_image(blur_frame)
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

command = ['ffmpeg-concat', '-T', "./templates/template1/input/transition.json",'-o', background_video, '-c 9'] + video_clip_names

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
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")

for file_name in os.listdir(temp_folder):
    file_path = os.path.join(temp_folder, file_name)

    if file_name != "background_video.mp4":
        os.remove(file_path)
