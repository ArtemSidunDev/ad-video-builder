# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:23:55 2025

@author: codemaven
"""

import cv2
import numpy as np
from PIL import Image, ImageOps
from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoClip, VideoFileClip, ImageClip, concatenate_videoclips
# import moviepy.audio.fx.all as afx
import subprocess
import os
import shutil
import argparse

# Load your image using PIL
VIDEO_FPS = 25
(DEST_WIDTH, DEST_HEIGHT) = (1080, 1920)

# Aspect ratios for comparison
ASPECT_RATIOS = {
    "Square": (1, 1),
    "Tall": (9, 16),
    "Wide": (4, 3),
    "Half": (9, 8)
}


parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
args = parser.parse_args()

folder_path = args.folderPath

TEMP_FOLDER = os.path.join(folder_path,"./temp/")


if not os.path.exists(TEMP_FOLDER):
    os.makedirs(TEMP_FOLDER)

def replace_green_background( foreground_media, replacement_media, t=None, fore_center_x=DEST_WIDTH//2, fore_center_y=DEST_HEIGHT//2):
    """
    Replaces the green background in a frame with another image.
    """
    if isinstance(foreground_media, VideoClip):
        frame = foreground_media.get_frame(t)
    elif isinstance(foreground_media, Image.Image):
        frame = np.array(foreground_media)
        
    if isinstance(replacement_media, VideoClip):
        replacement_frame = replacement_media.get_frame(t)
    elif isinstance(replacement_media, Image.Image):
        replacement_frame = np.array(replacement_media)
    
    rows, cols, _ = replacement_frame.shape
    background = replacement_frame.copy()
    
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

def pop_overlay_from_direction(t, duration, fore_media, back_media, stop_height=0, remove_green=False, pop_direction="bottom"):
    """
    Function to animate an image popping in from any direction in a video.
    
    :param t: Current time in seconds
    :param duration: Total duration of the effect
    :param fore_media: The foreground image or video
    :param back_media: The background image or video
    :param stop_height: The final position where the image should stop appearing
    :param remove_green: Decide if remove green color and make transparent 
    :param pop_direction: The direction from which the image should pop in. Default is "bottom".
    :return: Frame with the animated overlay
    """
    if t > duration:
        t = duration
    
    # Compute the new height or width based on the direction
    if pop_direction == "bottom" or pop_direction == "top":
        new_height = int((DEST_HEIGHT - stop_height) * (t / duration))
    elif pop_direction == "left" or pop_direction == "right":
        new_width = int((DEST_WIDTH - stop_height) * (t / duration))
    
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
        cropped_image = fore_image.crop((0, DEST_HEIGHT-new_height, DEST_WIDTH, DEST_HEIGHT))
    elif pop_direction == "right":
        cropped_image = fore_image.crop((0, 0, new_width, DEST_HEIGHT))
    elif pop_direction == "left":
        cropped_image = fore_image.crop((DEST_WIDTH-new_width, 0, DEST_WIDTH, DEST_HEIGHT))

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
        back_frame = Image.fromarray(back_media.get_frame(t))
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
        
    return back_frame.astype(int)

def slide_show(t, duration, back_media, slide_medias, offset_y, margin_x, speed=300, direction = "left"):
    if isinstance(back_media, VideoClip):
        back_frame = Image.fromarray(back_media.get_frame(t)).copy()
    elif isinstance(back_media, Image.Image):
        back_frame = back_media.copy()
    
    num_products = len(slide_medias)
    total_product_width = sum(media.width for media in slide_medias) + num_products * margin_x

    if direction == "right":
        scroll_position = (t * speed) % total_product_width
    elif direction == "left":
        scroll_position = -(t * speed) % total_product_width

    
    front_frames = []
    for i, product_media in enumerate(slide_medias):
        if isinstance(product_media, VideoClip):
            product_frame = Image.fromarray(product_media.get_frame(t))
        elif isinstance(product_media, Image.Image):
            product_frame = product_media
        x_pos = scroll_position - (i * (product_frame.width + margin_x))

        if x_pos > back_frame.width:
            x_pos -= total_product_width
        elif x_pos < -product_frame.width:
            x_pos += total_product_width
        
        front_frames.append((product_frame, x_pos))

    for front_frame, x_pos in front_frames:
        y_pos = offset_y
        back_frame.paste(front_frame, (int(x_pos), y_pos), front_frame.convert('RGBA'))
    return np.array(back_frame)

# =============================================================================
# # Global variable for use
# =============================================================================
 
avatar_video = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).without_audio().resize( ( int(1.1*DEST_WIDTH), int(1.1*DEST_HEIGHT))).crop( int(0.05*DEST_WIDTH), 0, int(1.05*DEST_WIDTH), DEST_HEIGHT)

avatar_back = Image.open(os.path.join(folder_path, "background_image.jpg")).convert("RGB").resize((DEST_WIDTH, DEST_HEIGHT))
green_back = Image.open(os.path.join("./templates/template5/input/greenscreen.png")).convert("RGB")

# =============================================================================
# =============================================================================
# # # Part 1. Concatenate videoclips 
# =============================================================================
# =============================================================================

# =============================================================================
# 1. Frame 0-110 (00:00 - 04:40)
# =============================================================================

time_start, time_end = 0.30, 4.42
duration = time_end - time_start

back_video = ImageClip(np.array(avatar_back), duration=duration)

# fore_image is a PNG with transparency, ensure it is loaded in RGBA mode
fore_image = resize_with_scaled_target(Image.open(os.path.join(folder_path, "1.png")), "Square").convert("RGBA").resize((DEST_WIDTH * 2 // 3, DEST_WIDTH * 2 // 3))
video_clip1 = VideoClip(lambda t: appear_with_effect(t, 0.28, duration-0.4, DEST_WIDTH // 2, DEST_HEIGHT * 2 // 5, back_video, fore_image), duration=duration-0.4)

video_clip2 = concatenate_videoclips([back_video.subclip( 0, time_start), video_clip1, back_video.subclip( 0, 0.4)], method="compose").subclip( 0, time_end)

fore_image2 = Image.open(os.path.join(folder_path, "textHookImage.png")).convert("RGBA").resize((DEST_WIDTH // 2, DEST_HEIGHT * 4 // 31))
duration2 = time_end

video_clip = VideoClip(lambda t: appear_with_effect(t, 0, duration2, DEST_WIDTH * 5 // 14, DEST_HEIGHT * 1 // 5, video_clip2, fore_image2), duration=duration2)
video_clip.subclip(0, duration2).write_videofile(f"{TEMP_FOLDER}01.mp4", fps=VIDEO_FPS, codec="libx264")

fore_image.close()
fore_image2.close()
back_video.close()
video_clip1.close()
video_clip2.close()
video_clip.close()

# =============================================================================
# 2. Frame 101-235 (04:02 - 09:30)
# =============================================================================

time_start, time_end = 4.02, 9.42
duration = time_end - time_start
video_clip = resize_with_scaled_target(VideoFileClip(os.path.join(folder_path, "ss.mp4")).without_audio().subclip( 0, duration), target="Tall").resize((DEST_WIDTH, DEST_HEIGHT))
video_clip.subclip( 0, duration).write_videofile(f"{TEMP_FOLDER}02.mp4", fps=VIDEO_FPS, codec="libx264")

# =============================================================================
# 1-2. Frame 101-110 (04:02 - 09:30) Transition
# =============================================================================

time_start, time_end = 4.00, 4.40
duration = time_end - time_start

command = f'ffmpeg -i 01.mp4 -i 02.mp4 -filter_complex "[0][1]xfade=transition=coverup:duration={duration}:offset={time_start},format=yuv420p" 01-02.mp4'

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
# 3. Frame 226-312
# =============================================================================

time_start, time_end = 9.06, 12.50
duration = time_end - time_start

image = Image.open(os.path.join(folder_path, "5.png")).convert("RGB")
scale = image.width * DEST_HEIGHT / (image.height * DEST_WIDTH)

video_clip = VideoClip(
    lambda t: sliding_frame(
        t,
        direction="left",
        image=image,
        video_dest_width=DEST_WIDTH,
        video_dest_height=DEST_HEIGHT,
        duration=duration,
        scale = scale,
        scale_direction = 'horizontal'
    ),
    duration=duration
)
video_clip.subclip( 0, duration).write_videofile(f"{TEMP_FOLDER}03.mp4", fps=VIDEO_FPS, codec="libx264")

# =============================================================================
# 1-2-3. Frame 226-235 (9.04 - 9.4) Transition
# =============================================================================

time_start, time_end = 9.04, 9.4
duration = time_end - time_start

command = f'ffmpeg -i 01-02.mp4 -i 03.mp4 -filter_complex "[0][1]xfade=transition=revealdown:duration={duration}:offset={time_start},format=yuv420p" 01-02-03-back.mp4'

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
# Add foreground video

time_start1, time_end1 = 0, 4.02
duration1 = time_end1 - time_start1

video_clip01_1 = avatar_video.subclip(0, duration1).resize( ( int( 1.1*DEST_WIDTH), int( 1.1*DEST_HEIGHT))).crop( int( 0.05*DEST_WIDTH), int( 0.05*DEST_HEIGHT), int( 1.05*DEST_WIDTH), int( 1.05*DEST_HEIGHT))


time_start2, time_end2 = 4.02, 4.42
duration2 = time_end2 - time_start2

video_clip01_2 = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video.subclip( time_start2, time_end2),
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration2,
               zoom_dest= 1.1  # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration2
       )
video_clip01_2.write_videofile(f"{TEMP_FOLDER}02-front.mp4", fps=VIDEO_FPS, codec="libx264")
# 
video_clip01 = concatenate_videoclips([video_clip01_1, VideoFileClip(f"{TEMP_FOLDER}02-front.mp4"), avatar_video.subclip( time_end2, 6.02)], method="compose").subclip( 0, 6.02)
back_video01 = VideoFileClip(f"{TEMP_FOLDER}01-02-03-back.mp4") 

######
time_start3, time_end3 = 0, 6.02
duration3 = time_end3 - time_start3

video_clip1 = VideoClip(lambda t: replace_green_background( foreground_media=video_clip01.resize( ( DEST_WIDTH*5//6, DEST_HEIGHT*5//6)),
                                                            replacement_media = back_video01.subclip(time_start3, time_end3), t=t, 
                                                            fore_center_x=DEST_WIDTH*9//12, fore_center_y=DEST_HEIGHT*7//12), 
                         duration=duration3)
video_clip1.subclip( 0, duration3).write_videofile(f"{TEMP_FOLDER}01-02-03-1.mp4", fps=VIDEO_FPS, codec="libx264")

time_start4, time_end4 = 6.02, 9.42
duration4 = time_end4 - time_start4

video_clip02 = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video.subclip( time_start4, time_end4),
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration4,
               zoom_dest= 0.85 # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration4
       ).subclip( 0, duration4)

video_clip2 = VideoClip(lambda t: replace_green_background( foreground_media=video_clip02.resize( ( DEST_WIDTH*5//6, DEST_HEIGHT*5//6)),
                                                            replacement_media = back_video01.subclip(time_start4, time_end4), t=t, 
                                                            fore_center_x=DEST_WIDTH*4//12, fore_center_y=DEST_HEIGHT*7//12), 
                         duration=duration4)
video_clip2.subclip( 0, duration4).write_videofile(f"{TEMP_FOLDER}01-02-03-2.mp4", fps=VIDEO_FPS, codec="libx264")

time_start5, time_end5 = 9.1, 9.28
duration5 = time_end5 - time_start5


video_clip3 = VideoClip(lambda t: appear_with_effect(t, duration5, 2 * duration5, DEST_WIDTH*4//12, DEST_HEIGHT*7//12, back_video01.subclip( time_start5-duration5, time_end5), 
                                                     video_clip02.resize( ( int( 0.98 * DEST_WIDTH), int(  0.98 * DEST_HEIGHT))).crop(DEST_WIDTH*2//12, DEST_HEIGHT//12, DEST_WIDTH*5//6, DEST_HEIGHT*11//12), inital_percentage=1, remove_green= True), 
                        duration=2 * duration5)
video_clip3.subclip( duration5, duration5*2).write_videofile(f"{TEMP_FOLDER}01-02-03-3.mp4", fps=VIDEO_FPS, codec="libx264")

video_clip = concatenate_videoclips([VideoFileClip(f"{TEMP_FOLDER}01-02-03-1.mp4"), VideoFileClip(f"{TEMP_FOLDER}01-02-03-2.mp4").subclip( 0, time_start5-time_start4), 
                                     VideoFileClip(f"{TEMP_FOLDER}01-02-03-3.mp4"), back_video01.subclip(time_end5, time_end4)], method="compose").subclip( 0, time_end)

video_clip.subclip(0, time_end4).write_videofile(f"{TEMP_FOLDER}01-02-03.mp4", fps=VIDEO_FPS, codec="libx264")

video_clip01_1.close()
video_clip01_2.close()
video_clip1.close()
video_clip02.close()
video_clip2.close()
video_clip3.close()

time_start, time_end = 9.42, 12.50
duration = time_end - time_start

fore_video = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video.subclip( time_start, time_end),
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration,
               zoom_dest= 0.75 # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration
       )
video_clip = VideoClip(lambda t: replace_green_background( foreground_media=fore_video.resize( ( DEST_WIDTH*5//6, DEST_HEIGHT*5//6)),
                                                            replacement_media = back_video01.subclip(time_start, time_end), t=t, 
                                                            fore_center_x=DEST_WIDTH*9//12, fore_center_y=DEST_HEIGHT*7//12), 
                         duration=duration)
video_clip.write_videofile(f"{TEMP_FOLDER}04.mp4", fps=VIDEO_FPS, codec="libx264")
video_clip.close()

# =============================================================================
# 5. Frame 312-479
# =============================================================================

time_start, time_end = 12.50, 19.18
duration = time_end - time_start

slide_medias = [
    resize_with_scaled_target( Image.open(os.path.join(folder_path, "1.png")), "Square").convert("RGBA"),
    resize_with_scaled_target( Image.open(os.path.join(folder_path, "2.png")), "Square").convert("RGBA"),
    resize_with_scaled_target( Image.open(os.path.join(folder_path, "3.png")), "Square").convert("RGBA")
]

fixed_height = int( DEST_HEIGHT / 3.13)

resized_slide_medias = []
for image in slide_medias:
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(fixed_height * aspect_ratio)
    resized_image = image.resize((new_width, fixed_height))
    resized_slide_medias.append(resized_image)
    
video_clip1 = VideoClip(lambda t: slide_show(t, duration, avatar_back, resized_slide_medias, int(DEST_HEIGHT / 6.33), 50, 300), duration=duration)

back_video = VideoClip(lambda t: appear_with_effect(t, 0.28, duration + 0.28, DEST_WIDTH//2, DEST_HEIGHT//2, avatar_back, video_clip1, inital_percentage=1), duration=duration+0.28)
# back_video.subclip( 0.28, duration+0.12).write_videofile(f"{TEMP_FOLDER}05.mp4", fps=VIDEO_FPS, codec="libx264")
back_video.subclip(duration+0.12, duration+0.28).write_videofile(f"{TEMP_FOLDER}06-01.mp4", fps=VIDEO_FPS, codec="libx264")

fore_video = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video.subclip( time_start + 0.28, time_end + 0.28),
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration - 0.16,
               zoom_dest= 0.75 # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration - 0.16
       )
video_clip = VideoClip(lambda t: replace_green_background( foreground_media=fore_video.resize( ( DEST_WIDTH*5//6, DEST_HEIGHT*5//6)),
                                                            replacement_media = back_video.subclip( 0.28, duration+0.12), t=t, 
                                                            fore_center_x=DEST_WIDTH*4//12, fore_center_y=DEST_HEIGHT*7//12), 
                         duration=duration - 0.16)

video_clip.write_videofile(f"{TEMP_FOLDER}05.mp4", fps=VIDEO_FPS, codec="libx264")

video_clip.close()
fore_video.close()
video_clip1.close()
back_video.close()
# =============================================================================


# =============================================================================
# 6. Frame 475-554 (19.02 - 19.82)
# =============================================================================

time_start, time_end = 19.02, 22.18
duration = time_end - time_start

back_video = concatenate_videoclips([VideoFileClip(f"{TEMP_FOLDER}06-01.mp4"), ImageClip(np.array(avatar_back), duration=duration - 0.16)], method="compose")

fore_image = resize_with_scaled_target( Image.open(os.path.join(folder_path, "7.png")), "Square").convert("RGBA").resize((DEST_WIDTH * 4 // 5, DEST_WIDTH * 4 // 5))
video_clip = VideoClip(lambda t: appear_with_effect(t, 0.4, duration, DEST_WIDTH // 2, DEST_HEIGHT // 3, back_video, fore_image, inital_percentage=0.1), duration=duration)
video_clip.subclip( 0, duration).write_videofile(f"{TEMP_FOLDER}06.mp4", fps=VIDEO_FPS, codec="libx264")

video_clip.close()
back_video.close()

# =============================================================================
# 7. Frame 544-623 (19.02 - 19.82)
# =============================================================================

time_start, time_end = 21.78, 24.94
duration = time_end - time_start

back_video = ImageClip(np.array(avatar_back), duration=duration)
fore_image = resize_with_scaled_target( Image.open(os.path.join(folder_path, "5.png")), "Square").convert("RGBA").resize((DEST_WIDTH*4//5, DEST_WIDTH*4//5 ))
video_clip = VideoClip(lambda t: appear_with_effect(t, 0.28, duration, DEST_WIDTH // 2, DEST_HEIGHT // 3, back_video, fore_image, inital_percentage=0.1), duration=duration)
video_clip.subclip( 0, duration).write_videofile(f"{TEMP_FOLDER}07.mp4", fps=VIDEO_FPS, codec="libx264")

video_clip.close()
back_video.close()

# =============================================================================
# 8. Frame 623 - the end
# =============================================================================

time_start, time_end = 25.36, avatar_video.duration
duration = time_end - time_start

fore_video = VideoClip(
           lambda t: zoom_frame(
               t,
               media=avatar_video.subclip( time_start, time_end).resize( ( DEST_WIDTH, DEST_HEIGHT)),
               video_dest_width=DEST_WIDTH,
               video_dest_height=DEST_HEIGHT,
               duration=duration,
               zoom_dest= 0.75 # Target zoom level for final frame (e.g., <1 for zoom out)
           ),
           duration=duration
       )

video_clip = VideoClip(lambda t: replace_green_background( foreground_media=fore_video, replacement_media = avatar_back, t=t), duration=duration)
video_clip.write_videofile(f"{TEMP_FOLDER}08.mp4", fps=VIDEO_FPS, codec="libx264")


video_clip_names = ['01-02-03.mp4', '04.mp4', '05.mp4', '06.mp4', '07.mp4', '08.mp4']
video_clips = [VideoFileClip(f"{TEMP_FOLDER}{clip_name}") for clip_name in video_clip_names]
final_video_without_subtitle = concatenate_videoclips(video_clips, method="compose")

# Write the final concatenated video to a new file
final_video_without_subtitle.write_videofile(f"{TEMP_FOLDER}concatenated_video.mp4", codec="libx264", fps=VIDEO_FPS)
