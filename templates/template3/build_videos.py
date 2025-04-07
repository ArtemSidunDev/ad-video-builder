# -*- coding: utf-8 -*-
"""
Created on Mon May  6 07:01:03 2024

@author: codemaven
"""

import os
import cv2
import copy
import numpy as np
from PIL import Image, ImageOps
from moviepy.editor import VideoClip, VideoFileClip, ImageClip, clips_array, concatenate_videoclips
import subprocess
import argparse

parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
args = parser.parse_args()

folder_path = args.folderPath

# Load your image using PIL
video_fps = 30
video_dest_width = 1216
video_dest_height = 2160
wipe_left_time = 400

speed_factor = 1.0

GREEN_COLOR = (15, 250, 74)

# Aspect ratios for comparison
ASPECT_RATIOS = {
    "Square": (1, 1),
    "Tall": (9, 16),
    "Wide": (4, 3),
    "Half": (9, 8)
}

video_spans = [ 2, 2, 3, 4, 4, 3, 3, 3, 6.7, 4.3]
transition_spans = [0.6, 0.6, 0, 0.4, 0.6, 0.6, 0.4, 0.4, 0.6]

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

def add_foreground(frame, t, foreground_clip, foreground_x, foreground_y, foreground_width, foreground_height, zoom_dest=1.5, offset_y=100):
    if foreground_x is None or foreground_y is None:
        return frame
    
    front_frame = foreground_clip.get_frame(t)
    zoomed_frame = build_zoomed_avatar( t = t, avatar_frame = front_frame, zoom_dest=zoom_dest, offset_y=offset_y)
    frame = replace_green_background(zoomed_frame, frame, t, fore_center_x= foreground_x + foreground_width//2, fore_center_y = foreground_y + foreground_height//2)
    return frame

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
    def _moving_average(curve, radius):
        window_size = 2 * radius + 1
        f = np.ones(window_size)/window_size
        curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        return curve_smoothed[radius:-radius]
    
    image_width, image_height = image.size

    # Calculate the zoom factor based on the zoom destination
    zoom_factor =  (t / duration) - (t / (zoom_dest * duration)) + (1 / zoom_dest) if zoom_dest > 1 else 1 - (1 - zoom_dest) * (t / duration)
    
    # Apply moving average for stabilization
    zoom_factor_smoothed = _moving_average(np.array([zoom_factor]), radius=5)[0]
    # Calculate zoom factor based on whether zooming in or out
    # if zoom_dest > 1:  # Zooming in to the original size
    #     zoom_factor = 1 / (zoom_dest - ((zoom_dest - 1) * (t / duration)**0.5))
    # elif zoom_dest < 1:  # Zooming out from the original size
    #     zoom_factor = 1 - ((1 - zoom_dest) * (t / duration)**0.5)
    # else:
    #     zoom_factor = 1  # No zoom effect if zoom_dest is 1

    # Calculate new dimensions for the zoomed area
    new_width = int(image_width * zoom_factor_smoothed)
    new_height = int(image_height * zoom_factor_smoothed)

    # Center-calculate cropping coordinates
    left = (image_width - new_width) // 2
    top = (image_height - new_height) // 2
    right = left + new_width
    bottom = top + new_height

    # Crop and resize the image to fit video destination dimensions
    cropped_image = image.crop((left, top, right, bottom))
    resized_image = cropped_image.resize((video_dest_width, video_dest_height), Image.Resampling.LANCZOS)
    
    return np.array(resized_image)

# Updated zoom_frame function with stabilization
# def zoom_and_move_frame(t):
#     global transition_span, zoom_dest, dest_width, dest_height, image_width, image_height, is_to_right
    
#     def _moving_average(curve, radius):
#         window_size = 2 * radius + 1
#         f = np.ones(window_size)/window_size
#         curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
#         curve_smoothed = np.convolve(curve_pad, f, mode='same')
#         return curve_smoothed[radius:-radius]
    
#     new_left = int((image_width - video_dest_width) * t / transition_span) # Moves from right to left
#     if is_to_right:
#         new_left = image_width - video_dest_width - new_left
    
#     zoom_image = src_image.crop( ( new_left, 0, new_left + dest_width, dest_height))
#     croped_width, croped_height = zoom_image.size
    
#     # Calculate the zoom factor based on the zoom destination
#     zoom_factor =  (t / transition_span) - (t / (zoom_dest * transition_span)) + (1 / zoom_dest) if zoom_dest > 1 else 1 - (1 - zoom_dest) * (t / transition_span)
    
#     # Apply moving average for stabilization
#     zoom_factor_smoothed = _moving_average(np.array([zoom_factor]), radius=20)[0]
    
#     # Calculate new dimensions
#     new_width = int(croped_width * zoom_factor_smoothed)
#     new_height = int(croped_height * zoom_factor_smoothed)
    
#     # Center the cropped area
#     left = (croped_width - new_width) / 2
#     top = (croped_height - new_height) / 2
#     right = left + new_width
#     bottom = top + new_height
    
#     # Crop and resize the image
#     cropped_image = zoom_image.crop((left, top, right, bottom))
#     resized_image = cropped_image.resize((dest_width, dest_height), Image.Resampling.LANCZOS)
    
#     return np.array(resized_image)

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
def get_closest_aspect_ratio(image, video_dest_width=video_dest_width, video_dest_height=video_dest_height):
    """
    Returns the closest predefined aspect ratio to the given image dimensions, excluding 'Half'.
    
    Parameters:
    - image (PIL.Image): The input image.
    - video_dest_width (int): The width of the destination video.
    - video_dest_height (int): The height of the destination video.
    
    Returns:
    - str: The name of the closest aspect ratio ('Square', 'Tall', or 'Wide').
    - float: horizontal scale.
    - float: vertical scale.
    """
    # Get image dimensions
    image_width, image_height = image.size

    # Calculate the aspect ratio of the image
    image_aspect_ratio = image_width / image_height

    # Filter out 'Half' from the ASPECT_RATIOS for comparison
    filtered_aspect_ratios = {key: value for key, value in ASPECT_RATIOS.items() if key != "Half"}

    # Find the closest match by calculating the difference with each target aspect ratio
    closest_ratio_name, closest_ratio = min(
        filtered_aspect_ratios.items(),
        key=lambda ratio: abs(image_aspect_ratio - ratio[1][0] / ratio[1][1])
    )

    # Calculate width and height scales based on the destination video dimensions
    return closest_ratio_name, closest_ratio

def get_aspect_ratio_conversion(source_name, target_name):
    """
    Calculate the horizontal and vertical ratios needed to convert one aspect ratio to another.
    
    Parameters:
    - source_name (str): The name of the source aspect ratio (e.g., "Square").
    - target_name (str): The name of the target aspect ratio (e.g., "Tall").
    
    Returns:
    - dict: A dictionary with horizontal and vertical ratios.
    """
    source_ratio = ASPECT_RATIOS[source_name]
    target_ratio = ASPECT_RATIOS[target_name]
    
    source_width, source_height = source_ratio
    target_width, target_height = target_ratio

    # Calculate the horizontal and vertical scaling ratios
    horizontal_ratio = target_width / source_width
    vertical_ratio = target_height / source_height

    # Adjust ratios to match the aspect without distortion (crop or fit)
    if horizontal_ratio < vertical_ratio:
        # Fit to height, crop width
        horizontal_ratio =  source_width * target_height/ (source_height * target_width)  
        vertical_ratio = 1
    else:
        # Fit to width, crop height
        vertical_ratio = source_height * target_width / (source_width * target_height)
        horizontal_ratio = 1

    return horizontal_ratio, vertical_ratio

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

def sliding_frame(t, direction, image, video_dest_width, video_dest_height, duration, scale=1.0, scale_direction="horizontal"):
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
    - scale: total margin/real width scale for slide
    - scale_direction: Direction for sclae("horizontal", "vertical")
    Returns:
    - np.array: The resized and cropped image frame for the current time t.
    """
    if scale_direction == "horizontal":
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
    
        back_frame[ int(offset_y*zoom_dest):, :] = resized_image[ 0 : int(new_height - offset_y*zoom_dest), :]   
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

# =================== GLOBAL Variable Declearation ============================
# global transition_span, zoom_dest, dest_width, dest_height, image_width, image_height, is_to_right, foreground_x, foreground_y
# ===================== 1. Zoom and Moving Left ===============================
# Time: 0-2s

clip_length = get_video_length( idx=1)
clip_start, clip_end = get_video_timespan( idx=1)

image = Image.open(os.path.join(folder_path, "1.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)
# original_width, original_height = image.size
# src_image = image.resize( ( int( original_width * video_dest_height / original_height) , video_dest_height))

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((video_dest_width, video_dest_height))
    processed_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height,
                duration=clip_length,
                zoom_dest= 0.9  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=clip_length
        )
elif closest_ratio_name == "Square" or closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    processed_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="left",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=clip_length,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=clip_length
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      

    
blur_duration = get_transition_span( idx = 1)
clip_duration = clip_length
blurred_clip = processed_clip.fl(apply_increasing_blur)

blurred_clip.subclip( 0, clip_length).write_videofile(f"{temp_folder}/01.mp4", codec="libx264", fps=video_fps)

image.close()
# src_image.close()
processed_clip.close()
blurred_clip.close()

# ===================== 2. Zoom and Moving Right ==============================
# Time: 2-4s

image = Image.open(os.path.join(folder_path, "2.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

clip_length = get_video_length( idx=2)
clip_start, clip_end = get_video_timespan( idx=2)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((video_dest_width, video_dest_height))
    video_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height,
                duration=clip_length,
                zoom_dest= 1.2  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=clip_length
        )
elif closest_ratio_name == "Square" or closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=clip_length,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=clip_length
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      
 
blur_duration = get_transition_span( idx = 1)
blurred_clip = video_clip.fl(apply_decreasing_blur)
# Write the processed video to a file
blurred_clip.subclip( 0, clip_length).write_videofile( f"{temp_folder}/02.mp4", codec="libx264", fps=video_fps)

image.close()
# src_image.close()
video_clip.close()
blurred_clip.close()

# Concatenate video 1 and video 2
command = f"ffmpeg-concat -t SimpleZoom -d {int(transition_spans[0]*1000)} -o 01-02-back.mp4 01.mp4 02.mp4"

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
    
foreground_clip = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( 0, background_clip.duration).resize((video_dest_width*0.75, video_dest_height*0.75))

foreground_width, foreground_height = foreground_clip.size[0], foreground_clip.size[1]
foreground_x, foreground_y = ( video_dest_width - foreground_width)//2, video_dest_height - foreground_height

processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t, foreground_clip, foreground_x, foreground_y, foreground_width, foreground_height))

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

background_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).subclip(clip_start, clip_end)
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

image = Image.open(os.path.join(folder_path, "3.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

clip_length = get_video_length( idx=4)
clip_start, clip_end = get_video_timespan( idx=4)

background_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).subclip(clip_start, clip_end)

if closest_ratio_name == "Square":
    adjusted_image = resize_with_scaled_target(image, target="Tall").resize((video_dest_width * 2 // 5, video_dest_height * 2 // 5))
    fore_clip1  = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width * 2 // 5,
                video_dest_height=video_dest_height * 2 // 5,
                duration=clip_length,
                zoom_dest= 0.7  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=clip_length
        )
elif closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name, scale=1.2, scale_direction="height")
    _,vertical_scale = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    fore_clip1  = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="down",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width * 2 // 5,
            video_dest_height=video_dest_height * 2 // 5,
            duration=clip_length,
            scale = vertical_scale * 1.2,
            scale_direction = 'vertical'
        ),
        duration=clip_length
    )
elif closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    fore_clip1  = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="left",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width * 2 // 5,
            video_dest_height=video_dest_height * 2 // 5,
            duration=clip_length,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=clip_length
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      

fore_clip1.write_videofile(f"{temp_folder}/04-1.mp4", codec="libx264", fps=video_fps)

image.close()
fore_clip1.close()

image = Image.open(os.path.join(folder_path, "4.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

if closest_ratio_name == "Square":
    adjusted_image = resize_with_scaled_target(image, target="Tall").resize((video_dest_width * 2 // 5, video_dest_height * 2 // 5))
    fore_clip2  = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width * 2 // 5,
                video_dest_height=video_dest_height * 2 // 5,
                duration=clip_length,
                zoom_dest= 0.7  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=clip_length
        )
elif closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name, scale=1.2, scale_direction="height")
    _,vertical_scale = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    fore_clip2  = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="down",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width * 2 // 5,
            video_dest_height=video_dest_height * 2 // 5,
            duration=clip_length,
            scale = 1.2 * vertical_scale,
            scale_direction = 'vertical'
        ),
        duration=clip_length
    )
elif closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    fore_clip2  = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width * 2 // 5,
            video_dest_height=video_dest_height * 2 // 5,
            duration=clip_length,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=clip_length
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      

fore_clip2.write_videofile(f"{temp_folder}/04-2.mp4", codec="libx264", fps=video_fps)

image.close()
adjusted_image.close()
fore_clip1.close()

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
image = Image.open(os.path.join(folder_path, "5.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

clip_length = get_video_length( idx=5)
clip_start, clip_end = get_video_timespan( idx=5)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((video_dest_width, video_dest_height))
    background_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height,
                duration=clip_length,
                zoom_dest= 1.2  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=clip_length
        )
elif closest_ratio_name == "Square" or closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    background_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="left",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=clip_length,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=clip_length
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")     

foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).subclip( clip_start, clip_end).resize((video_dest_width*0.6, video_dest_height*0.6))

foreground_width, foreground_height = foreground_clip.size[0], foreground_clip.size[1]
foreground_x, foreground_y = video_dest_width - foreground_width+100, video_dest_height - foreground_height-50

processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t, foreground_clip, foreground_x, foreground_y, foreground_width, foreground_height))

blur_duration = get_transition_span( idx = 4)
blurred_start_clip = processed_clip.fl(apply_decreasing_blur)

blur_duration = get_transition_span( idx = 5)
clip_duration = clip_length
blurred_clip = blurred_start_clip.fl(apply_increasing_blur)
# Write the processed video to a file
blurred_clip.subclip( 0, clip_duration).write_videofile( f"{temp_folder}/05.mp4", codec="libx264", fps=video_fps)

image.close()
adjusted_image.close()
processed_clip.close()
blurred_start_clip.close()
blurred_clip.close()

# ===================== 6. Zoomming In Video ==================================
# Time: 15-18s

image = Image.open(os.path.join(folder_path, "6.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

clip_length = get_video_length( idx=6)
clip_start, clip_end = get_video_timespan( idx=6)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((video_dest_width, video_dest_height))
    processed_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height,
                duration=clip_length,
                zoom_dest= 0.8  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=clip_length
        )
elif closest_ratio_name == "Square" or closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    processed_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=clip_length,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=clip_length
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")     


blur_duration = get_transition_span( idx = 5)
blurred_start_clip = processed_clip.fl(apply_decreasing_blur)

blur_duration = get_transition_span( idx = 6)
clip_duration = clip_length
blurred_clip = blurred_start_clip.fl(apply_increasing_blur)

# Write the processed video to a file
blurred_clip.subclip( 0, clip_length).write_videofile( f"{temp_folder}/06.mp4", codec="libx264", fps=video_fps)

image.close()
adjusted_image.close()
processed_clip.close()
blurred_start_clip.close()
blurred_clip.close()

# ===================== 7. Zoomming Out Video ==================================
# Time: 18-21

image = Image.open(os.path.join(folder_path, "7.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

clip_length = get_video_length( idx=7)
clip_start, clip_end = get_video_timespan( idx=7)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((video_dest_width, video_dest_height))
    processed_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height,
                duration=clip_length,
                zoom_dest= 1.2  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=clip_length
        )
elif closest_ratio_name == "Square" or closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    processed_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="left",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height,
            duration=clip_length,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=clip_length
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")     


# processed_clip.write_videofile(f"{temp_folder}/07.mp4", codec="libx264", fps=video_fps)

blur_duration = get_transition_span( idx = 6)
blurred_clip = processed_clip.fl(apply_decreasing_blur)

blurred_clip.crop(0, 0, video_dest_width, video_dest_height//2).write_videofile(f"{temp_folder}/07-1.mp4", codec="libx264", fps=video_fps)
blurred_clip.crop(0, video_dest_height//2, video_dest_width, video_dest_height).write_videofile(f"{temp_folder}/07-2.mp4", codec="libx264", fps=video_fps)

image.close()
adjusted_image.close()
processed_clip.close()
blurred_clip.close()

# ===================== 8. Moving Left and Right Videos =======================
# Time: 21-24

clip_length = get_video_length( idx=8)
clip_start, clip_end = get_video_timespan( idx=8)

image = Image.open(os.path.join(folder_path, "8.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

if closest_ratio_name == "Square":
    adjusted_image = resize_with_scaled_target(image, target="Half").resize((video_dest_width, video_dest_height //2))
    upper_video_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height//2,
                duration=clip_length,
                zoom_dest= 1.3  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=clip_length
        )
elif closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    _,vertical_scale = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    upper_video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="down",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=clip_length,
            scale = vertical_scale,
            scale_direction = 'vertical'
        ),
        duration=clip_length
    )
elif closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    upper_video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="left",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=clip_length,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=clip_length
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      


upper_video_clip.write_videofile(f"{temp_folder}/08-1.mp4", codec="libx264", fps=video_fps)
# video_clip.close()

image.close()
adjusted_image.close()


image = Image.open(os.path.join(folder_path, "9.png")).convert("RGB")
if closest_ratio_name == "Square":
    adjusted_image = resize_with_scaled_target(image, target="Half").resize((video_dest_width, video_dest_height //2))
    lower_video_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height//2,
                duration=clip_length,
                zoom_dest= 0.8  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=clip_length
        )
elif closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    _,vertical_scale = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    lower_video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="up",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=clip_length,
            scale = vertical_scale,
            scale_direction = 'vertical'
        ),
        duration=clip_length
    )
elif closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    lower_video_clip = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=clip_length,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=clip_length
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      

lower_video_clip.write_videofile(f"{temp_folder}/08-2.mp4", codec="libx264", fps=video_fps)
# video_clip.close()


image.close()
adjusted_image.close()
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
clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4"))
clip_length = get_video_length( idx=9)
clip_start, clip_end = get_video_timespan( idx=9)

clip_end = clip.duration

avatar_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).subclip( clip_start, clip_end)

blur_duration = get_transition_span( idx = 8)
blurred_start_clip = avatar_clip.fl(apply_decreasing_blur)

blur_duration = get_transition_span( idx = 9)
clip_duration = avatar_clip.duration
blurred_clip = blurred_start_clip.fl(apply_increasing_blur)

blurred_clip.write_videofile(f"{temp_folder}/09.mp4", temp_audiofile=f"{temp_folder}/09.mp3", remove_temp=True,codec="libx264", fps=video_fps)
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

clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).subclip(8, 8+clip_length)
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
                                          , background_clip.subclip( background_clip.duration - foreground_clip.duration, background_clip.duration).fl(lambda gf, t: add_foreground(gf(t), t, foreground_clip, foreground_x, foreground_y, foreground_width, foreground_height))])

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

background_video = "background_video.mp4"
command = ['ffmpeg-concat', '-T', "../../../templates/template3/input/transition.json",
            '-o', background_video] + video_clip_names
try:
    completed_process = subprocess.run(
        " ".join(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
        cwd=temp_folder)
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

processed_clip = background_clip.subclip(15, 21).fl(lambda gf, t: add_foreground(gf(t), t, foreground_clip, foreground_x, foreground_y, foreground_width, foreground_height))
final_clip = concatenate_videoclips([background_clip.subclip(0, 15), processed_clip, background_clip.subclip(21, background_clip.duration)])
final_clip.write_videofile( f"{temp_folder}/overlapped_video.mp4", codec="libx264", fps=video_fps)

for file_name in os.listdir(temp_folder):
    file_path = os.path.join(temp_folder, file_name)

    if file_name != "overlapped_video.mp4":
        os.remove(file_path)