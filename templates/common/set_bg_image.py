import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import argparse

parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
args = parser.parse_args()

folder_path = args.folderPath

def remove_green_background(frame, background_image):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    mask = cv2.medianBlur(mask, 11)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    mask_inv = cv2.bitwise_not(mask)

    fg = cv2.bitwise_and(frame, frame, mask=mask_inv)

    background_resized = cv2.resize(background_image, (frame.shape[1], frame.shape[0]))

    bg = cv2.bitwise_and(background_resized, background_resized, mask=mask)

    combined = cv2.add(fg, bg)

    return combined

def process_video(video_path, background_image_path, output_path):
    video = VideoFileClip(video_path)
    background_image = cv2.imread(background_image_path)
    
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    
    new_video = video.fl_image(lambda frame: remove_green_background(frame, background_image))

    new_video.write_videofile(output_path, temp_audiofile=f"{folder_path}/bg_avatar.mp3", remove_temp=True, codec='libx264', fps=30)

process_video(f"{folder_path}/avatar.mp4", f"{folder_path}/background_image.jpg", f"{folder_path}/bg_avatar.mp4")