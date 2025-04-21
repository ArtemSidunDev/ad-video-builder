# -*- coding: utf-8 -*-
"""
Updated on Feb 1 07:01:03 2024

@author: codemaven
"""

import os
import cv2
import copy
import numpy as np
from PIL import Image, ImageOps
from moviepy.editor import VideoClip, VideoFileClip, ImageClip, clips_array, concatenate_videoclips
import subprocess
import os
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

video_spans = [ 2, 2, 3, 4, 3, 2, 5, 5, 4, 5]
transition_spans = [0.4, 0.4, 0, 0.6, 0.4, 0, 0.6, 0.4, 0.4]

temp_folder = os.path.join(folder_path, "temp")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    
def add_foreground(frame, t, zoom_dest=1.2, offset_y=0):
    global foreground_clip, foreground_x, foreground_y, foreground_width, foreground_height
    front_frame = foreground_clip.get_frame(t)
    # return frame
    if foreground_x is None or foreground_y is None:
        return frame

    zoomed_frame = build_zoomed_avatar( t = t, avatar_frame = front_frame, zoom_dest=zoom_dest, offset_y=offset_y)
    frame = replace_green_background(zoomed_frame, frame, t, fore_center_x= foreground_x + foreground_width//2, fore_center_y = foreground_y + foreground_height//2)
    return frame

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

global transition_span, zoom_dest, dest_width, dest_height, zoom_image
global foreground_clip, foreground_width, foreground_height, foreground_x, foreground_y  

# ================ 1. ZoomOut(Top) / avatar(Bottom) ===========================
# Time 0-2s
# Load the image and convert it to RGB
image = Image.open(os.path.join(folder_path, "1.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

duration = get_video_length( idx=1)
clip_start, clip_end = get_video_timespan( idx=1)

if closest_ratio_name == "Square":
    adjusted_image = resize_with_scaled_target(image, target="Half").resize((video_dest_width, video_dest_height //2))
    upper_video_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height//2,
                duration=duration,
                zoom_dest= 1.3  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=duration
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
            duration=duration,
            scale = vertical_scale,
            scale_direction = 'vertical'
        ),
        duration=duration
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
            duration=duration,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=duration
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      

lower_video_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).without_audio().subclip(clip_start, clip_end).crop(
    int(0.1*video_dest_width), video_dest_height // 4, int(0.9*video_dest_width), int( video_dest_height // 4 + video_dest_height * 0.8 / 2 )).resize((video_dest_width, video_dest_height//2))

# Stack the videos vertically
composed_clip = clips_array([[upper_video_clip], [lower_video_clip]])

blur_duration = get_transition_span( idx = 1)
clip_duration = duration
blurred_clip = composed_clip.fl(apply_increasing_blur)

blurred_clip.subclip( 0, duration).write_videofile(f"{temp_folder}/01.mp4", codec="libx264", fps=video_fps)

image.close()
upper_video_clip.close()
lower_video_clip.close()
composed_clip.close()
blurred_clip.close()

# ================ 2. ZoomIn + avatar(Left) ===================================
# Time 2-4s

# Load the image and convert it to RGB
image = Image.open(os.path.join(folder_path, "2.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

duration = get_video_length( idx=2)
clip_start, clip_end = get_video_timespan( idx=2)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((video_dest_width, video_dest_height))
    background_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height,
                duration=duration,
                zoom_dest= 0.9  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=duration
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
            duration=duration,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=duration
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      

foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).without_audio().subclip( clip_start, clip_end)

# Set foreground speacker size to 60% of background video
foreground_width = int(background_clip.size[0] * 0.7)
foreground_height = int( foreground_clip.size[1] * foreground_width / foreground_clip.size[0])
foreground_clip = foreground_clip.resize((foreground_width, foreground_height))

foreground_x, foreground_y = -80, video_dest_height - foreground_height
# Apply the replace_green_background function to each frame of the
# Deley for 2 seconds
processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t, offset_y=100))

blur_duration = get_transition_span( idx = 1)
blurred_clip = processed_clip.fl(apply_decreasing_blur)
# Write the processed video to a file
blurred_clip.subclip( 0, duration).write_videofile( f"{temp_folder}/02.mp4", codec="libx264", fps=video_fps)

image.close()
background_clip.close()
foreground_clip.close()
processed_clip.close()
blurred_clip.close()

# ================ 3. ScreenCast + avatar(Center) =============================
# Time 4-7s
# video starts from 5s
clip_length = get_video_length( idx=3)
clip_start, clip_end = get_video_timespan( idx=3)

clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).without_audio().subclip(5, 5 + clip_length)
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

foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).without_audio().subclip( clip_start, clip_end).crop(
    int(0.1*video_dest_width), video_dest_height // 4, int(0.9*video_dest_width), int( video_dest_height // 4 + video_dest_height * 0.8 / 2 )).resize((video_dest_width, video_dest_height//2))

# foreground_clip.write_videofile("output.mp4", fps=video_fps, codec="libx264")
foreground_width, foreground_height = foreground_clip.size[0], foreground_clip.size[1]
foreground_x, foreground_y = ( video_dest_width - foreground_width)//2, video_dest_height // 2

processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t, zoom_dest=1.0))

# Write the processed video to a file
processed_clip.subclip(0, clip_length).write_videofile( f"{temp_folder}/03.mp4", codec="libx264", fps=video_fps)

clip.close()
cropped_clip.close()
foreground_clip.close()
background_clip.close()
processed_clip.close()

# ================ 4. Moving left + avatar( Top-Center) =======================
# Time 7-9s, 9-11s

clip_length = get_video_length( idx=4)
clip_start, clip_end = get_video_timespan( idx=4)

clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).without_audio().subclip(8, 5 + clip_length)
aspect_ratio = clip.w / clip.h
dest_ratio = video_dest_width / video_dest_height

if aspect_ratio > dest_ratio:
    new_width = int(clip.h * dest_ratio)
    new_height = clip.h
else:
    new_width = clip.w
    new_height = int(clip.w / dest_ratio)

upper_video_clip = clip.crop( 0,  102,  new_width, 102 + new_height).resize( (video_dest_width, video_dest_height))

image = Image.open(os.path.join(folder_path, "3.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)
duration = clip_length/2 + 0.2

if closest_ratio_name == "Square":
    adjusted_image = resize_with_scaled_target(image, target="Half").resize((video_dest_width, video_dest_height //2))
    lower_video_clip1 = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height//2,
                duration=duration,
                zoom_dest= 1.5  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=duration
        )
elif closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    _,vertical_scale = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    lower_video_clip1 = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="down",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=duration,
            scale = vertical_scale,
            scale_direction = 'vertical'
        ),
        duration=duration
    )
elif closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    lower_video_clip1 = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="left",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=duration,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=duration
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      
    
blur_duration = 0.4
clip_duration = duration
lower_blurred_clip1 = lower_video_clip1.fl(apply_increasing_blur)
lower_blurred_clip1.write_videofile(f"{temp_folder}/04-1.mp4", codec="libx264", fps=video_fps)

image.close()
# move_image.close()

image = Image.open(os.path.join(folder_path, "4.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)
duration = clip_length/2 + 0.2

if closest_ratio_name == "Square":
    adjusted_image = resize_with_scaled_target(image, target="Half").resize((video_dest_width, video_dest_height //2))
    lower_video_clip2 = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height//2,
                duration=duration,
                zoom_dest= 0.67  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=duration
        )
elif closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    _,vertical_scale = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    lower_video_clip2 = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="up",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=duration,
            scale = vertical_scale,
            scale_direction = 'vertical'
        ),
        duration=duration
    )
elif closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    lower_video_clip2 = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=duration,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=duration
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")   

blur_duration = 0.4
clip_duration = duration
lower_blurred_clip2 = lower_video_clip2.fl(apply_decreasing_blur)
lower_blurred_clip2.write_videofile(f"{temp_folder}/04-2.mp4", codec="libx264", fps=video_fps)

command = "ffmpeg-concat -T ../../../templates/template2/input/zoomin_transition.json -o 04_lower.mp4 04-1.mp4 04-2.mp4"

try:
    completed_process = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
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

lower_video_clip = VideoFileClip(f"{temp_folder}/04_lower.mp4").without_audio().subclip( 0, clip_length)

# add trasition
def pop_from_bottom(t):
    if t <= trans_duration:
        new_height = int(video_dest_height * t // (2 * trans_duration))
    else:
        new_height = video_dest_height // 2 
    global foreground_x, foreground_y, foreground_width, foreground_height
    foreground_width, foreground_height = int(foreground_clip.size[0]), int( foreground_clip.size[1])
    foreground_x = ( video_dest_width - foreground_width)//2
    foreground_y = video_dest_height // 2 - new_height    
    lower_frame = lower_video_clip.get_frame(t)
    upper_frame = upper_video_clip.get_frame(t)
        # Crop the lower frame using slicing
    cropped_image = lower_frame[:new_height, :video_dest_width]
    
    # Paste the cropped image onto the upper frame
    upper_frame[video_dest_height - new_height:, :video_dest_width] = cropped_image
    return add_foreground( upper_frame, t, zoom_dest=1)
    # return upper_frame

trans_duration = 1/4
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).without_audio().subclip( clip_start ,clip_end).crop(
    int(0.1*video_dest_width), video_dest_height // 4, int(0.9*video_dest_width), int( video_dest_height // 4 + video_dest_height * 0.8 / 2 )).resize((video_dest_width, video_dest_height//2))

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

# ================ 5. ZoomIn + avatar( Left) ==================================
# Time 11-14s

clip_length = get_video_length( idx=5)
clip_start, clip_end = get_video_timespan( idx=5)

image = Image.open(os.path.join(folder_path, "5.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((video_dest_width, video_dest_height))
    background_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height,
                duration=clip_length,
                zoom_dest= 0.9  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=duration
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
        duration=duration
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      

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

image.close()
background_clip.close()
foreground_clip.close()
processed_clip.close()
blurred_clip.close()

# ================ 6. Full screen avatar video ================================
# Time 14-16s

clip_length = get_video_length( idx=6)
clip_start, clip_end = get_video_timespan( idx=6)

avatar_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).without_audio().subclip(clip_start ,clip_end)
avatar_clip.write_videofile(f"{temp_folder}/06.mp4", codec="libx264", fps=video_fps)
avatar_clip.close()

# ================ 7. Full screen avatar video ================================
# Time 16-18.5s, 18.5-21

clip_length = get_video_length( idx=7)
clip_start, clip_end = get_video_timespan( idx=7)
transition_span = clip_length/2 + 0.2

image = Image.open(os.path.join(folder_path, "6.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)
if closest_ratio_name == "Square":
    adjusted_image = resize_with_scaled_target(image, target="Half").resize((video_dest_width, video_dest_height //2))
    upper_video_clip1 = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height//2,
                duration=transition_span,
                zoom_dest= 0.9  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=duration
        )
elif closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    _,vertical_scale = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    upper_video_clip1 = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="down",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=transition_span,
            scale = vertical_scale,
            scale_direction = 'vertical'
        ),
        duration=duration
    )
elif closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    upper_video_clip1 = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="left",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=transition_span,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=duration
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")    

blur_duration = 0.4
clip_duration = transition_span
upper_blurred_clip1 = upper_video_clip1.fl(apply_increasing_blur)
upper_blurred_clip1.write_videofile(f"{temp_folder}/07-1.mp4", fps=video_fps)

image.close()

# for 7_2
image = Image.open(os.path.join(folder_path, "7.png")).convert("RGB")
closest_ratio_name, _ = get_closest_aspect_ratio( image)

transition_span = clip_length/2 + 0.2

if closest_ratio_name == "Square":
    adjusted_image = resize_with_scaled_target(image, target="Half").resize((video_dest_width, video_dest_height //2))
    upper_video_clip2 = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height//2,
                duration=transition_span,
                zoom_dest= 1.1  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=duration
        )
elif closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    _,vertical_scale = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    upper_video_clip2 = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="up",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=transition_span,
            scale = vertical_scale,
            scale_direction = 'vertical'
        ),
        duration=duration
    )
elif closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Half")
    upper_video_clip2 = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=video_dest_width,
            video_dest_height=video_dest_height//2,
            duration=transition_span,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=duration
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      


blur_duration = 0.4
clip_duration = transition_span
upper_blurred_clip2 = upper_video_clip2.fl(apply_decreasing_blur)

upper_blurred_clip2.write_videofile(f"{temp_folder}/07-2.mp4", fps=video_fps)

# command = f'"C:\\Program Files\\FFmpeg\\ffmpeg.exe" -i 07-1.mp4 -i 07-2.mp4 -filter_complex "[0][1]xfade=transition=zoomin:duration=0.4:offset={clip_length/2-0.2},format=yuv420p" 07_upper.mp4'
command = "ffmpeg-concat -T ../../../templates/template2/input/zoomin_transition.json -o 07_upper.mp4 07-1.mp4 07-2.mp4"

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

upper_video_clip = VideoFileClip(f"{temp_folder}/07_upper.mp4").without_audio().subclip( 0, clip_length)

lower_video_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).without_audio().subclip( clip_start, clip_end).crop( 0, video_dest_height//6, video_dest_width, video_dest_height//2 + video_dest_height//6)

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

back_frame = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).without_audio().get_frame( clip_start).copy()

transition_span = 1/4

back_frame_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).without_audio().subclip( clip_start, clip_start + transition_span)
processed_clip = VideoClip(drop_from_top, duration=5)

blur_duration = get_transition_span( idx = 7)
clip_duration = clip_length
blurred_clip = processed_clip.fl(apply_increasing_blur)

blurred_clip.subclip(0, clip_length).write_videofile(f"{temp_folder}/07.mp4", codec="libx264", fps=video_fps)

image.close()
# zoom_image.close()
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
closest_ratio_name, _ = get_closest_aspect_ratio( image)

transition_span = clip_length

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((video_dest_width, video_dest_height))
    background_clip = VideoClip(
            lambda t: zoom_frame(
                t,
                image=adjusted_image,
                video_dest_width=video_dest_width,
                video_dest_height=video_dest_height,
                duration=transition_span,
                zoom_dest= 0.9  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=transition_span
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
            duration=transition_span,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=transition_span
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      
# Create the video clip using the modified zoom_in_frame function
foreground_clip = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).without_audio().subclip( clip_start, clip_end)

# Set foreground speacker size to 70% of background video
foreground_width = int(background_clip.size[0] * 0.7)
foreground_height = int( foreground_clip.size[1] * foreground_width / foreground_clip.size[0])
foreground_clip = foreground_clip.resize((foreground_width, foreground_height))

foreground_x, foreground_y = video_dest_width - foreground_width + 80, video_dest_height - foreground_height
# Apply the replace_green_background function to each frame of the
processed_clip = background_clip.fl(lambda gf, t: add_foreground(gf(t), t, offset_y=100))

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

avatar_clip = VideoFileClip(os.path.join(folder_path, "bg_avatar.mp4")).without_audio().subclip( clip_start, clip_end)
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

clip = VideoFileClip(os.path.join(folder_path, "ss.mp4")).without_audio().subclip(10, 10+clip_length)
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

foreground_clip = VideoFileClip(os.path.join(folder_path, "action.mp4")).without_audio().subclip( 0, clip_length - action_delay)

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
