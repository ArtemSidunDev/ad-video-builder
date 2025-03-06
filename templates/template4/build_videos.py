# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:23:55 2025

@author: codemaven
"""
import gc
import cv2
import numpy as np
from PIL import Image, ImageOps
from moviepy.editor import AudioFileClip, CompositeAudioClip,VideoClip, VideoFileClip, ImageClip, concatenate_videoclips, clips_array
import moviepy.audio.fx.all as afx
import subprocess
import os
import shutil
import copy
import argparse

# Load your image using PIL
VIDEO_FPS = 25
(DEST_WIDTH, DEST_HEIGHT) = (1080, 1920)

GREEN_COLOR = (15, 250, 74)
# Aspect ratios for comparison
ASPECT_RATIOS = {
    "Square": (1, 1),
    "Tall": (9, 16),
    "Wide": (4, 3),
    "Half": (9, 8)
}

parser = argparse.ArgumentParser(description="Generate a video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
args = parser.parse_args()

folder_path = args.folderPath
# folder_path = "./input/"

TEMP_FOLDER = os.path.join(folder_path,"./temp/")
# INPUT_FOLDER = 

if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

def replace_green_background( foreground_media, replacement_media, t=None, fore_center_x=DEST_WIDTH//2, fore_center_y=DEST_HEIGHT//2):
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
    del frame
    del replacement_frame
    gc.collect()
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

def pop_overlay_from_direction(t, duration, fore_media, back_media, stop_position=0, remove_green=False, pop_direction="bottom"):
    """
    Function to animate an image popping in from any direction in a video.
    
    :param t: Current time in seconds
    :param duration: Total duration of the effect
    :param fore_media: The foreground image or video
    :param back_media: The background image or video
    :param stop_position: The final position where the image should stop appearing
    :param remove_green: Decide if remove green color and make transparent 
    :param pop_direction: The direction from which the image should pop in. Default is "bottom".
    :return: Frame with the animated overlay
    """
    if t > duration:
        t = duration
    
    # Compute the new height or width based on the direction
    if pop_direction == "bottom" or pop_direction == "top":
        new_height = int((DEST_HEIGHT - stop_position) * (t / duration))
    elif pop_direction == "left" or pop_direction == "right":
        new_width = int((DEST_WIDTH - stop_position) * (t / duration))
    
    if isinstance(back_media, VideoClip):
        back_frame = back_media.get_frame(t)
        back_image = Image.fromarray(back_frame)
    else:
        back_image = back_media
        
    if isinstance(fore_media, VideoClip):
        fore_frame = fore_media.get_frame(t)
        if remove_green:
            fore_frame = remove_green_background(fore_frame)
        fore_image = Image.fromarray(fore_frame)
    else:
        fore_image = fore_media.convert("RGBA") if remove_green else fore_media
    
    # Resize image to fit video width and crop it progressively
    if pop_direction == "bottom":
        cropped_image = fore_image.crop((0, 0, DEST_WIDTH, new_height))
    elif pop_direction == "top":
        cropped_image = fore_image.crop((0, fore_image.height-new_height, DEST_WIDTH, fore_image.height))
    elif pop_direction == "right":
        cropped_image = fore_image.crop((0, 0, new_width, DEST_HEIGHT))
    elif pop_direction == "left":
        cropped_image = fore_image.crop((fore_image.width-new_width, 0, fore_image.width, DEST_HEIGHT))

    if remove_green:
        back_image = back_image.convert("RGBA")        
        if pop_direction == "bottom":
            back_image.paste(cropped_image, (0, DEST_HEIGHT - new_height), cropped_image)
        elif pop_direction == "top":
            back_image.paste(cropped_image, (0, 0), cropped_image)
        elif pop_direction == "left":
            back_image.paste(cropped_image, (0, 0), cropped_image)
        elif pop_direction == "right":
            back_image.paste(cropped_image, (DEST_WIDTH - new_width, 0), cropped_image)
        return np.array(back_image.convert("RGB"))
    else:
        if pop_direction == "bottom":
            back_image.paste(cropped_image, (0, DEST_HEIGHT - new_height))
        elif pop_direction == "top":
            back_image.paste(cropped_image, (0, 0))
        elif pop_direction == "left":
            back_image.paste(cropped_image, (0, 0))
        elif pop_direction == "right":
            back_image.paste(cropped_image, (DEST_WIDTH - new_width, 0))
        return np.array(back_image)

def zoom_frame(t, media, video_dest_width, video_dest_height, duration, zoom_dest=1.0):
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
        curve_pad = np.pad(curve, (radius, radius), 'edge')
        curve_smoothed = np.convolve(curve_pad, f, mode='same')
        return curve_smoothed[radius:-radius]
    
    image_width, image_height = media.size
    
    if isinstance(media, VideoClip):
        frame = media.get_frame(t)
        image = Image.fromarray(frame)
    else:
        image = media

    # Calculate the zoom factor based on the zoom destination
    zoom_factor =  (t / duration) - (t / (zoom_dest * duration)) + (1 / zoom_dest) if zoom_dest > 1 else 1 - (1 - zoom_dest) * (t / duration)
    
    # Apply moving average for stabilization
    zoom_factor_smoothed = _moving_average(np.array([zoom_factor]), radius=5)[0]

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

def get_closest_aspect_ratio(image, video_dest_width=DEST_WIDTH, video_dest_height=DEST_HEIGHT):
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

def apply_split_or_merge(t, front_media, back_media, duration, video_dest_width=DEST_WIDTH, video_dest_height=DEST_HEIGHT, effect_type="split"):

    if isinstance(front_media, VideoClip):
        front_frame = Image.fromarray(front_media.get_frame(t))
    elif isinstance(front_media, Image.Image):
        front_frame = front_media
        
    if isinstance(back_media, VideoClip):
        if t < back_media.duration:
            back_frame = Image.fromarray(back_media.get_frame(t).copy())
        else:
            back_frame = Image.fromarray(back_media.get_frame(back_media.duration))
    elif isinstance(back_media, Image.Image):
        back_frame = back_media

    if effect_type == "split":
        new_height = int(video_dest_height * (duration - t) // (2 * duration))
        top_image = front_frame.crop(
            (0, video_dest_height // 2 - new_height, video_dest_width, video_dest_height // 2))
        bottom_image = front_frame.crop(
            (0, video_dest_height // 2, video_dest_width, video_dest_height // 2 + new_height))
        
        back_frame.paste(top_image, (0, 0))
        back_frame.paste(bottom_image, (0, video_dest_height - new_height))
        
        image = np.array(back_frame)
        region_box = (0, new_height, video_dest_width, video_dest_height - new_height)
    
    elif effect_type == "merge":
        new_height = int(video_dest_height * t // (2 * duration))
        top_image = front_frame.crop(
            (0, 0, video_dest_width, new_height))
        bottom_image = front_frame.crop(
            (0, video_dest_height - new_height, video_dest_width, video_dest_height))

        back_frame.paste(top_image, (0, 0))
        back_frame.paste(bottom_image, (0, video_dest_height - new_height))
        
        image = np.array(back_frame)
        region_box = (0, new_height, video_dest_width, video_dest_height - new_height)
    
    # Apply blur to the region of interest
    max_blur_strength = 20
    blur_step = 4
    blurred_image = image.copy()

    for i in range(0, max_blur_strength, blur_step):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (region_box[0], region_box[1] + (max_blur_strength - i)),
                      (region_box[2], region_box[3] - (max_blur_strength - i)), 255, -1)

        blurred_image_copy = blurred_image.copy()
        blurred_image_copy = cv2.GaussianBlur(blurred_image_copy, (int(20 * t) * 2 + 1, int(20 * t) * 2 + 1), i)
        if effect_type == "split":
            blurred_image = np.where(mask[..., None] != 0, blurred_image, blurred_image_copy)
        else:
            blurred_image = np.where(mask[..., None] == 0, blurred_image, blurred_image_copy)
                                         
    return blurred_image

def appear_with_effect(t, trans_duration, total_duration, center_x, center_y, back_media, fore_media, inital_percentage = 0.4, remove_green = False):
    if isinstance(back_media, Image.Image):
        back_frame = np.array(back_media).copy() 
    else:
        back_frame = back_media.get_frame(t).copy()
    
    if isinstance(fore_media, Image.Image):
        fore_frame = np.array(fore_media.convert("RGBA"))
    else:
        fore_frame = fore_media.get_frame(t)
        alpha_channel = np.ones((fore_frame.shape[0], fore_frame.shape[1]), dtype=fore_frame.dtype) * 255
        if remove_green:        
            hsv = cv2.cvtColor(fore_frame, cv2.COLOR_RGB2HSV)
            lower_green = np.array([40, 40, 40])  # Lower bound of green color
            upper_green = np.array([70, 255, 255])  # Upper bound of green color
            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            # Set the alpha channel to 0 (transparent) where green pixels are found
            alpha_channel[mask == 255] = 0
            
        fore_frame = np.dstack((fore_frame, alpha_channel))
        
    fore_height, fore_width = fore_frame.shape[:2]

    if t < trans_duration:
        fore_width = int(fore_width * inital_percentage + fore_width * (1 - inital_percentage) * t / trans_duration)
        fore_height = int(fore_height * inital_percentage + fore_height * (1 - inital_percentage) * t / trans_duration)
    elif t > total_duration - trans_duration:
        fore_width = int(fore_width * inital_percentage + (1 - inital_percentage) * fore_width * (total_duration - t) / trans_duration)
        fore_height = int(fore_height * inital_percentage + (1 - inital_percentage) * fore_height * (total_duration - t) / trans_duration)

    left = center_x - fore_width // 2
    top = center_y - fore_height // 2
    right = center_x + fore_width // 2
    bottom = center_y + fore_height // 2

    # Resize the foreground to fit the calculated dimensions
    resized_fore_frame = cv2.resize(fore_frame, (right - left, bottom - top))

    # Handle the blending of the foreground and background with transparency
    if t < trans_duration:
        alpha = t / trans_duration
    elif t > total_duration - trans_duration:
        alpha = (total_duration - t) / trans_duration
    else:
        alpha = 1.0

    for c in range(3):  # RGB channels
        back_frame[top:bottom, left:right, c] = back_frame[top:bottom, left:right, c] * (1 - alpha * resized_fore_frame[:, :, 3] / 255) + resized_fore_frame[:, :, c] * (alpha * resized_fore_frame[:, :, 3] / 255)
        
    return back_frame

def build_zoomed_avatar(t, avatar_video, zoom_dest = 1.0, offset_y = 0, back_media=None):
    """
    Creates a zoomed avatar video with a replaced green background.
    """
    
    avatar_frame = avatar_video.get_frame(t)
    
    # Zoom the avatar frame
    image_height, image_width, _ = avatar_frame.shape
    new_width, new_height = int(image_width / zoom_dest), int(image_height / zoom_dest)
    
    if back_media == None:
        back_frame = np.full((image_height, image_width, 3), GREEN_COLOR, dtype=np.uint8)
    elif isinstance(back_media, VideoClip):
        back_frame = back_media.get_frame(t)
    elif isinstance(back_media, Image.Image):
        back_frame = np.array(back_media)
    elif isinstance(back_media, np.ndarray):
        back_frame = back_media

    if zoom_dest >= 1:
        left = (image_width - new_width) // 2
        top = (image_height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
    
        cropped_image = avatar_frame[top:bottom, left:right]
        resized_image = np.array(Image.fromarray(cropped_image).resize((image_width, image_height), Image.Resampling.LANCZOS))
    
        back_frame[offset_y:, :] = resized_image[0:image_height - offset_y, :]   
    else:
        new_width = int(image_width * zoom_dest)
        new_height = int(image_height * zoom_dest)
        
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
    
# =============================================================================
# # Global variable for use
# =============================================================================
# avatar_back = Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))
green_back_image = Image.fromarray( np.full((DEST_HEIGHT, DEST_WIDTH, 3), GREEN_COLOR, dtype=np.uint8))


# avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw, avatar_back, t), duration=avatar_video_raw.duration)
# avatar_back.close()
# =============================================================================
# 1. 00.00 - 00.20 Avatar appears from the bottom
# =============================================================================

time_start, time_end = 0, 0.22
duration = time_end - time_start

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)

video_clip1 = VideoClip(lambda t: pop_overlay_from_direction(t, duration, avatar_video_raw_zoomed, green_back_image), duration=duration)
back_image = Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))
video_clip = VideoClip(lambda t: replace_green_background(video_clip1, back_image, t), duration=duration)
video_clip.write_videofile(f"{TEMP_FOLDER}01.mp4", fps=VIDEO_FPS, codec="libx264")

video_clip.close()
video_clip1.close()
avatar_video_raw_zoomed.close()
avatar_video_raw.close()
del back_image

# =============================================================================
# 2. 00.20 - 02.20 Smooth increase in avatar size (~+10%)
# =============================================================================

time_start, time_end = 0.22, 2.70
# time_start, time_end = 0.22, 2.50
duration = time_end - time_start

back_image = Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image, t), duration=avatar_video_raw_zoomed.duration)
video_clip2 = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video,
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration,
               zoom_dest= 115/125  # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration
       )

video_clip2.write_videofile(f"{TEMP_FOLDER}02.mp4", fps=VIDEO_FPS, codec="libx264")
video_clip2.close()
avatar_video_raw.close()
avatar_video_raw_zoomed.close()
avatar_video.close()

# =============================================================================
# 3. 01.80 - 3.08 Frame 68-82 fast decrease in avatar size ~+10%
# =============================================================================

# 3.1
time_start, time_end = 2.70, 3.02
duration = time_end - time_start
# avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw, back_image, t), duration=avatar_video_raw.duration)
# 
avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)
back_image= Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image, t), duration=avatar_video_raw.duration)

video_clip = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video,
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration,
               zoom_dest= 125/115  # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration
       )
video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}03-1.mp4", codec="libx264", fps=VIDEO_FPS)

video_clip.close()
avatar_video_raw_zoomed.close()
avatar_video.close()
avatar_video_raw.close()


# 3.2
time_start, time_end = 3.02, 3.30
duration = time_end - time_start

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)
back_image = Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image, t), duration=avatar_video_raw.duration)

video_clip = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video,
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration,
               zoom_dest= 115/140 # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration
       )
video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}03-2.mp4", codec="libx264", fps=VIDEO_FPS)

video_clip.close()
avatar_video_raw_zoomed.close()
avatar_video.close()
avatar_video_raw.close()

# =============================================================================
# 4.Frame 83-108, 109-118(transition) zoom in -10% to original
# =============================================================================
time_start, time_end = 3.30, 4.74
duration = time_end - time_start

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.25, offset_y=150), duration=duration)
back_image_zoomed = Image.fromarray( zoom_frame(0, Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT)), DEST_WIDTH, DEST_HEIGHT, 1, 125/115))
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image_zoomed, t), duration=avatar_video_raw.duration)

video_clip = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video,
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration,
               zoom_dest= 140/125  # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration
       )
video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}04.mp4", codec="libx264", fps=VIDEO_FPS)

video_clip.close()
avatar_video_raw_zoomed.close()
avatar_video.close()
avatar_video_raw.close()

video_clip = concatenate_videoclips([VideoFileClip(f"{TEMP_FOLDER}01.mp4"), VideoFileClip(f"{TEMP_FOLDER}02.mp4"),
                                     VideoFileClip(f"{TEMP_FOLDER}03-1.mp4"), VideoFileClip(f"{TEMP_FOLDER}03-2.mp4"), VideoFileClip(f"{TEMP_FOLDER}04.mp4")], method="compose")
video_clip.subclip(0, time_end).write_videofile(f"{TEMP_FOLDER}01-04.mp4", codec="libx264", fps=VIDEO_FPS)
video_clip.close()

# =============================================================================
# 5. 108 - 225
# =============================================================================
# Frame 108 - 118 color fill #F3EDE2
time_start1, time_end1 = 4.34, 5.06
duration1 = time_end1 - time_start1

initial_back_image = Image.new("RGB", (DEST_WIDTH, DEST_HEIGHT), (243, 237, 226))


time_start, time_end = 5.06, 9.02
duration = time_end - time_start

image_clip_names = [ 
         os.path.join(folder_path, "1-bordered.png"),
         os.path.join(folder_path, "2-bordered.png"),
         os.path.join(folder_path, "3-bordered.png")
    ]
rotates = [5, -3, 2]
direction = ["right", "left", "right"]

animation_duration = 0.4
stable_duration = 0.8

back_image = initial_back_image.copy()
for idx in range(len(image_clip_names)):
    image_clip =  Image.open(f"{image_clip_names[idx]}").convert("RGBA");
    new_width = int(0.65 * DEST_WIDTH)
    aspect_ratio = image_clip.height / image_clip.width
    new_height = int(new_width * aspect_ratio)
    
    image_clip = image_clip.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
      
    rotated_image = image_clip.rotate(rotates[idx], resample=Image.Resampling.BILINEAR, expand=True)
    
    # Calculate the new size for the background to prevent information loss
    dest_width = int( rotated_image.size[0] * 1.1)
    dest_height = int( DEST_HEIGHT * 1.1//2)  
    
    # Create a new RGBA image with the new size and a transparent background
    new_image = Image.new("RGBA", (dest_width, dest_height), (243, 237, 226, 0))
    
    # Calculate the position to center the rotated image in the new image
    x_offset = (dest_width - rotated_image.size[0]) // 2
    y_offset = int( 0.08 * DEST_HEIGHT)
    
    # Paste the rotated image onto the new image
    new_image.paste(rotated_image, (x_offset, y_offset), rotated_image)
    
    video_clip1 = VideoClip(lambda t: pop_overlay_from_direction(t, animation_duration, new_image, back_image, stop_position=0, remove_green=True, pop_direction=direction[idx]), duration=animation_duration)
    video_clip1.write_videofile(f"{TEMP_FOLDER}05_temp_{idx+1}.mp4", fps=VIDEO_FPS, codec="libx264")
    
    last_image = Image.fromarray( VideoFileClip(f"{TEMP_FOLDER}05_temp_{idx+1}.mp4").get_frame( video_clip1.duration-0.02))
    video_clip = concatenate_videoclips( [ video_clip1, ImageClip(np.array(last_image), duration=stable_duration)])
    video_clip.write_videofile(f"{TEMP_FOLDER}05_{idx+1}.mp4", fps=VIDEO_FPS, codec="libx264")
    
    back_image.close()
    back_image = last_image.copy()
    video_clip1.close()
    image_clip.close()
    new_image.close()
    last_image.close()

back_clip = concatenate_videoclips( [ImageClip(np.array(initial_back_image), duration=duration1), VideoFileClip(f"{TEMP_FOLDER}05_1.mp4"), VideoFileClip(f"{TEMP_FOLDER}05_2.mp4"), VideoFileClip(f"{TEMP_FOLDER}05_3.mp4"), ImageClip(np.array(back_image), duration=0.70)])
back_clip.write_videofile(f"{TEMP_FOLDER}05_back.mp4", fps=VIDEO_FPS, codec="libx264")

# appear 118-126
# total 118 - 225
trans_duration = 0.32
total_duration = 4.28
appear_start_time = 0.4

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip(4.72, 9.4).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
video_clip3 = VideoClip(lambda t: appear_with_effect(t, trans_duration=trans_duration, total_duration=total_duration+trans_duration, center_x = DEST_WIDTH//2, center_y=DEST_HEIGHT*3//5,
                                                     back_media=back_clip.subclip( appear_start_time, back_clip.duration), 
                                                     fore_media=avatar_video_raw.crop( 0, 0, DEST_WIDTH, 4*DEST_HEIGHT//5), 
                                                     inital_percentage=1, remove_green= True),
                        duration = total_duration+trans_duration).subclip(0, total_duration)
video_clip = concatenate_videoclips( [back_clip.subclip(0, appear_start_time), video_clip3.subclip(0, total_duration)])
video_clip.subclip(0, total_duration).write_videofile(f"{TEMP_FOLDER}05.mp4", fps=VIDEO_FPS, codec="libx264")

transition_start, transition_end = 4.32, 4.72
transition_duration = transition_end - transition_start

command = f'ffmpeg -i 01-04.mp4 -i 05.mp4 -filter_complex "[0][1]xfade=transition=coverleft:duration={transition_duration}:offset={transition_start},format=yuv420p" 01-05.mp4'
  
# Execute each command in the temp folder
try:
    completed_process = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
        cwd=TEMP_FOLDER
    )
    if completed_process.returncode == 0:
        print("Command output:")
        print(completed_process.stdout)
        print("Command executed successfully.")
    else:
        print(f"command includes errors:  {completed_process.stderr}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e.stderr}")

# =============================================================================
# 6. 215-225 (08:15-09:00)
# =============================================================================

time_start, time_end = 8.62, 9.02
duration = time_end - time_start

back_video = video_clip.copy().subclip(video_clip.duration-duration-0.02, video_clip.duration)
video_clip.close()

top_image = resize_with_scaled_target(Image.open( os.path.join(folder_path, "4.png")).convert("RGB"), target="Half").resize((DEST_WIDTH, DEST_HEIGHT//2))
bottom_image = resize_with_scaled_target(Image.open( os.path.join(folder_path, "5.png")).convert("RGB"), target="Half").resize((DEST_WIDTH, DEST_HEIGHT//2))
front_image = Image.fromarray(np.concatenate((np.array(top_image), np.array(bottom_image)), axis=0))

# Apply the desired effect: "split" or "merge"
video_clip = VideoClip(lambda t: apply_split_or_merge(t=t, front_media=front_image, back_media = back_video, duration= duration, effect_type = "merge"), duration=duration)

video_clip.write_videofile(f"{TEMP_FOLDER}06.mp4", codec="libx264", fps=VIDEO_FPS)
video_clip.close()

# =============================================================================
# 7. Frame 225-252 : Photos remain static
# =============================================================================

time_start, time_end = 9.02, 10.10
duration = time_end - time_start

video_clip = ImageClip(np.array(front_image), duration=duration)
video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}07.mp4", codec="libx264", fps=VIDEO_FPS)
video_clip.close()

# =============================================================================
# 8. Frame 252-263: Photos move upwards and downwards accordingly,
# =============================================================================

time_start, time_end = 10.10, 10.54
duration = time_end - time_start

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)
back_image = Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image, t), duration=avatar_video_raw.duration)

video_clip = VideoClip(lambda t: apply_split_or_merge(t, front_image, avatar_video, duration), duration=duration)
video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}08.mp4", codec="libx264", fps=VIDEO_FPS)

video_clip.close()
avatar_video.close()
avatar_video_raw_zoomed.close()
avatar_video_raw.close()
# =============================================================================
# 9. Frame 263-305: Smooth zoom-in on the avatar ~+15%
# =============================================================================

time_start, time_end = 10.54, 12.22
duration = time_end - time_start

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)
back_image = Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image, t), duration=avatar_video_raw.duration)

video_clip = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video,
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration,
               zoom_dest= 115/130  # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration
       )

video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}09.mp4", codec="libx264", fps=VIDEO_FPS)

video_clip.close()
avatar_video.close()
avatar_video_raw_zoomed.close()
avatar_video_raw.close()
# =============================================================================
# 10. Frame 380 (15:05): Cut transition
# =============================================================================

time_start, time_end = 12.22, 15.22
duration = time_end - time_start

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)
back_image = Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image, t), duration=avatar_video_raw.duration)

video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}10.mp4", codec="libx264", fps=VIDEO_FPS)

video_clip.close()
avatar_video.close()
avatar_video_raw_zoomed.close()
avatar_video_raw.close()
# =============================================================================
# 11. Frame 380-449 Background (the photo of the product) smoothly slides to the left
# =============================================================================

time_start, time_end = 15.22, 17.98
duration = time_end - time_start

image = Image.open( os.path.join(folder_path, "5.png")).convert("RGB")

closest_ratio_name, _ = get_closest_aspect_ratio( image)


if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((DEST_WIDTH, DEST_HEIGHT))
    back_video = VideoClip(
            lambda t: zoom_frame(
                t,
                media=adjusted_image,
                video_dest_width=DEST_WIDTH,
                video_dest_height=DEST_HEIGHT,
                duration=duration,
                zoom_dest= 1.1  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=duration
        )
elif closest_ratio_name == "Square" or closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    back_video = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=DEST_WIDTH,
            video_dest_height=DEST_HEIGHT,
            duration=duration,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=duration
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      


# horizontal_scale = image.width * DEST_HEIGHT / (image.height * DEST_WIDTH)
# back_video = VideoClip(
#     lambda t: sliding_frame(
#         t,
#         direction="right",  # 'left', 'up', or 'down' as needed
#         image=image,
#         video_dest_width=DEST_WIDTH,
#         video_dest_height=DEST_HEIGHT,
#         duration=duration,
#         scale = horizontal_scale,
#         scale_direction = 'horizontal'
#     ),
#     duration=duration
# )
  
avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize( ( DEST_WIDTH*5//6, DEST_HEIGHT*5//6))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)

video_clip = VideoClip(lambda t: replace_green_background( foreground_media=avatar_video_raw_zoomed,
                                                            replacement_media = back_video, t=t, 
                                                            fore_center_x=DEST_WIDTH*4//12, fore_center_y=DEST_HEIGHT*7//12), 
                        duration=duration
                    )
video_clip.crop(0, 0, DEST_WIDTH, DEST_HEIGHT//2).subclip(0, duration).write_videofile(f"{TEMP_FOLDER}11_top.mp4", codec="libx264", fps=VIDEO_FPS)
video_clip.crop(0, DEST_HEIGHT//2, DEST_WIDTH, DEST_HEIGHT).subclip(0, duration).write_videofile(f"{TEMP_FOLDER}11_bottom.mp4", codec="libx264", fps=VIDEO_FPS)

video_clip.close()
avatar_video_raw.close()
avatar_video_raw_zoomed.close()
# =============================================================================
# 12. Frame 439 - 483 (17:17): Cut transition
# =============================================================================

time_start, time_end = 17.58, 19.34
duration = time_end - time_start

top_image = resize_with_scaled_target(Image.open( os.path.join(folder_path, "4.png")).convert("RGB"), target="Half").resize((DEST_WIDTH, DEST_HEIGHT//2))
bottom_image = resize_with_scaled_target(Image.open( os.path.join(folder_path, "3.png")).convert("RGB"), target="Half").resize((DEST_WIDTH, DEST_HEIGHT//2))
front_image = Image.fromarray(np.concatenate((np.array(top_image), np.array(bottom_image)), axis=0))

video_clip = ImageClip(np.array(front_image), duration=duration)
video_clip.crop(0, 0, DEST_WIDTH, DEST_HEIGHT//2).subclip(0, duration).write_videofile(f"{TEMP_FOLDER}12_top.mp4", codec="libx264", fps=VIDEO_FPS)
video_clip.crop(0, DEST_HEIGHT//2, DEST_WIDTH, DEST_HEIGHT).subclip(0, duration).write_videofile(f"{TEMP_FOLDER}12_bottom.mp4", codec="libx264", fps=VIDEO_FPS)
video_clip.close()


# 11-12 transition 
time_start, time_end = 2.36, 2.76
duration = time_end - time_start

commands = [
        f'ffmpeg -i 11_top.mp4 -i 12_top.mp4 -filter_complex "[0][1]xfade=transition=coverleft:duration={duration}:offset={time_start},format=yuv420p" 11-12_top.mp4',
        f'ffmpeg -i 11_bottom.mp4 -i 12_bottom.mp4 -filter_complex "[0][1]xfade=transition=coverright:duration={duration}:offset={time_start},format=yuv420p" 11-12_bottom.mp4'
    ]

for command in commands:
    try:
        completed_process = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=True,
            text=True,
            cwd=TEMP_FOLDER
        )
        if completed_process.returncode == 0:
            print("Command output:")
            print(completed_process.stdout)
            print("Command executed successfully.")
        else:
            print(f"command includes errors:  {completed_process.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e.stderr}")

video_clip = clips_array([[VideoFileClip(f"{TEMP_FOLDER}11-12_top.mp4")], [VideoFileClip(f"{TEMP_FOLDER}11-12_bottom.mp4")]])
video_clip.write_videofile(f"{TEMP_FOLDER}11-12.mp4", codec="libx264", fps=VIDEO_FPS)
video_clip.close()
# =============================================================================
# 13. Frame 473-609 (19:08 - 24:09): Smooth zoom-in on the avatar ~+15%
# =============================================================================

time_start, time_end = 18.94, 24.38
duration = time_end - time_start

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)
back_image_zoomed = Image.fromarray( zoom_frame(0, Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT)), DEST_WIDTH, DEST_HEIGHT, 1, 1.553))
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image_zoomed, t), duration=avatar_video_raw.duration)

video_clip = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video,
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration,
               zoom_dest= 115/145 # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration
       )

video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}13.mp4", codec="libx264", fps=VIDEO_FPS)
video_clip.close()

# transition
time_start, time_end = 3.72, 4.12
duration = time_end - time_start

command = f'ffmpeg -i 11-12.mp4 -i 13.mp4 -filter_complex "[0][1]xfade=transition=slideleft:duration={duration}:offset={time_start},format=yuv420p" 11-13.mp4'

try:
    completed_process = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
        cwd=TEMP_FOLDER
    )
    if completed_process.returncode == 0:
        print("Command output:")
        print(completed_process.stdout)
        print("Command executed successfully.")
    else:
        print(f"command includes errors:  {completed_process.stderr}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e.stderr}")

# =============================================================================
# 14. Frame 609-614 Quick zoom-out to transition
# =============================================================================

time_start, time_end = 24.38, 24.58
duration = time_end - time_start

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)
back_image_zoomed = Image.fromarray( zoom_frame(0, Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT)), DEST_WIDTH, DEST_HEIGHT, 1, 1.553))
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image_zoomed, t), duration=avatar_video_raw.duration)

video_clip = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video,
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration,
               zoom_dest= 145/115  # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration
       )

video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}14.mp4", codec="libx264", fps=VIDEO_FPS)
video_clip.close()

# =============================================================================
# 15.Frame 614 -683, (24:15): Cut transition, avatar appears from the bottom, background - product photo
# =============================================================================
# 614 -683 background
time_start, time_end = 24.58, 27.34
duration = time_end - time_start

# Frame 614-683 background
image = Image.open( os.path.join(folder_path, "7.png")).convert("RGB")

closest_ratio_name, _ = get_closest_aspect_ratio( image)

if closest_ratio_name == "Tall":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name).resize((DEST_WIDTH, DEST_HEIGHT))
    back_video1 = VideoClip(
            lambda t: zoom_frame(
                t,
                media=adjusted_image,
                video_dest_width=DEST_WIDTH,
                video_dest_height=DEST_HEIGHT,
                duration=duration,
                zoom_dest= 0.9  # Target zoom level for final frame (e.g., <1 for zoom out)
            ),
            duration=duration
        )
elif closest_ratio_name == "Square" or closest_ratio_name == "Wide":
    adjusted_image = resize_with_scaled_target(image, target=closest_ratio_name)
    horizontal_scale,_ = get_aspect_ratio_conversion( source_name=closest_ratio_name, target_name="Tall")
    back_video1 = VideoClip(
        lambda t: sliding_frame(
            t,
            direction="right",  # 'left', 'up', or 'down' as needed
            image=adjusted_image,
            video_dest_width=DEST_WIDTH,
            video_dest_height=DEST_HEIGHT,
            duration=duration,
            scale = horizontal_scale,
            scale_direction = 'horizontal'
        ),
        duration=duration
    )
else:
    raise ValueError("Invalid target aspect ratio. Choose 'Square', 'Tall', or 'Wide'.")      

# scale = image.width * DEST_HEIGHT / (image.height * DEST_WIDTH)

# back_video1 = VideoClip(
#     lambda t: sliding_frame(
#         t,
#         direction="right",
#         image=image,
#         video_dest_width=DEST_WIDTH,
#         video_dest_height=DEST_HEIGHT,
#         duration=duration,
#         scale = scale,
#         scale_direction = 'horizontal'
#     ),
#     duration=duration
# )
back_video1.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}15-back_1.mp4", codec="libx264", fps=VIDEO_FPS)
back_video1.close()

# 675-683 background
time_start01, time_end01 = 27.02, 27.34
duration01 = time_end01 - time_start01

back_video2 = ImageClip( np.array(Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))), duration=duration01)
back_video2.subclip(0, duration01).write_videofile(f"{TEMP_FOLDER}15-back_2.mp4", codec="libx264", fps=VIDEO_FPS)
back_video2.close()
# slideleft

command = f'ffmpeg -i 15-back_1.mp4 -i 15-back_2.mp4 -filter_complex "[0][1]xfade=transition=slideleft:duration={duration01}:offset={time_start01-time_start},format=yuv420p" 15-back.mp4'

try:
    completed_process = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        text=True,
        cwd=TEMP_FOLDER
    )
    if completed_process.returncode == 0:
        print("Command output:")
        print(completed_process.stdout)
        print("Command executed successfully.")
    else:
        print(f"command includes errors:  {completed_process.stderr}")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing the command: {e.stderr}")

back_video = VideoFileClip(f"{TEMP_FOLDER}15-back.mp4")

# Frame 614-625 
time_start1, time_end1 = 24.58, 25.00
duration1 = time_end1 - time_start1

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start1, time_end1).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)

video_clip1 = VideoClip(lambda t: pop_overlay_from_direction(t, duration1, avatar_video_raw_zoomed, back_video.subclip(0, duration1), stop_position=0, remove_green=True), duration=duration1)

avatar_video_raw.close()
avatar_video_raw_zoomed.close()

# Frame 625-683
time_start2, time_end2 = 25.00, 27.36
duration2 = time_end2 - time_start2

avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).subclip( time_start2, time_end2).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)

fore_video = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video_raw_zoomed,
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration2,
               zoom_dest= 135/115  # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration2
       )

video_clip2 = VideoClip(lambda t: replace_green_background(fore_video, back_video.subclip(back_video.duration-duration2, back_video.duration), t), duration=duration2)

video_clip = concatenate_videoclips([video_clip1.subclip(0, duration1), video_clip2.subclip(0, duration2)])

video_clip.subclip(0, duration).write_videofile(f"{TEMP_FOLDER}15.mp4", codec="libx264", fps=VIDEO_FPS)

video_clip.close()
avatar_video_raw.close()
avatar_video_raw_zoomed.close()

# =============================================================================
# 16. 683-
# =============================================================================
avatar_video_raw = VideoFileClip( os.path.join(folder_path, "avatar.mp4"))

time_start, time_end = 27.34, avatar_video_raw.duration-0.04

avatar_video_raw = avatar_video_raw.subclip( time_start, time_end).without_audio().resize((DEST_WIDTH, DEST_HEIGHT))
avatar_video_raw_zoomed = VideoClip(lambda t: build_zoomed_avatar( t = t, avatar_video = avatar_video_raw, zoom_dest=1.15, offset_y=150), duration=duration)
back_image_zoomed = Image.fromarray( zoom_frame(0, Image.open( os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT)), DEST_WIDTH, DEST_HEIGHT, 1, 1.418))
avatar_video = VideoClip(lambda t: replace_green_background(avatar_video_raw_zoomed, back_image_zoomed, t), duration=avatar_video_raw.duration)


avatar_video.write_videofile(f"{TEMP_FOLDER}16.mp4", codec="libx264", fps=VIDEO_FPS)
avatar_video.close()
avatar_video_raw_zoomed.close()
avatar_video_raw.close()

# video_clips = [VideoFileClip(f"{TEMP_FOLDER}15.mp4"), VideoFileClip(f"{TEMP_FOLDER}16.mp4")]
# video_clip = concatenate_videoclips(video_clips, method="compose")
# video_clip.write_videofile(f"{TEMP_FOLDER}15-16.mp4", codec="libx264", fps=VIDEO_FPS)
# =============================================================================
# # Concatenate all videos
# =============================================================================

video_clip_names = ['01-05.mp4', '06.mp4', '07.mp4','08.mp4', '09.mp4', '10.mp4', '11-13.mp4', '14.mp4', '15.mp4', '16.mp4']
video_clips = [VideoFileClip(f"{TEMP_FOLDER}{clip_name}") for clip_name in video_clip_names]
final_video_without_subtitle = concatenate_videoclips(video_clips, method="compose")

# Write the final concatenated video to a new file
final_video_without_subtitle.write_videofile(f"{TEMP_FOLDER}concatenated_video.mp4", codec="libx264", fps=VIDEO_FPS)


# =============================================================================
# # # Part 2. Add avatar 
# =============================================================================
# =============================================================================
# =============================================================================
# 
# # concatenated_video = VideoFileClip( f"{TEMP_FOLDER}concatenated_video.mp4")
# subtitle_video = VideoFileClip( os.path.join(folder_path, "subtitles.mp4")).resize((DEST_WIDTH, DEST_HEIGHT))
# final_video = VideoClip(lambda t: replace_green_background( foreground_media=subtitle_video.without_audio(), replacement_media = final_video_without_subtitle, t=t), duration=final_video_without_subtitle.duration)
# foreground_audio = VideoFileClip( os.path.join(folder_path, "avatar.mp4")).audio
# # foreground_audio.write_audiofile(f"{TEMP_FOLDER}foreground_audio.mp3", codec='mp3')
# 
# 
# swoosh_audio = AudioFileClip( os.path.join(folder_path, "Swoosh 01 - 01.mp3"))
# slide_audio = AudioFileClip( os.path.join(folder_path, "Soft Slide 02 - Short 02.mp3"))
# 
# audios = [foreground_audio.fx(afx.audio_normalize).fx(afx.volumex, 1.5)]
# audios.append( AudioFileClip( os.path.join(folder_path, "ES_I_m Sweet (Instrumental Version) - Adelyn Paik (1).mp3")).subclip(0, foreground_audio.duration).fx(afx.audio_normalize).fx(afx.volumex, 0.2))
# 
# audios.append(swoosh_audio.set_start(108/25))
# audios.append(swoosh_audio.set_start(127/25))
# audios.append(swoosh_audio.set_start(157/25))
# audios.append(swoosh_audio.set_start(187/25))
# audios.append(slide_audio.set_start(217/25))
# audios.append(slide_audio.set_start(251/25))
# audios.append(slide_audio.set_start(441/25))
# audios.append(swoosh_audio.set_start(477/25))
# audios.append(swoosh_audio.set_start(612/25))
# audios.append(swoosh_audio.set_start(679/25))
# 
# composite_audio = CompositeAudioClip(audios)
# final_video = final_video.set_audio(composite_audio)
# 
# final_video.write_videofile("output.mp4", codec="libx264", fps=VIDEO_FPS, audio_codec="aac")
# final_video.close()
# 
# try:
#     shutil.rmtree(TEMP_FOLDER)
#     print(f"Folder '{TEMP_FOLDER}' deleted successfully.")
# except OSError as e:
#     print(f"Error: {e}")
# =============================================================================
