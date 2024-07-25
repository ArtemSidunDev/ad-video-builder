import cv2
import numpy as np
from PIL import Image
from moviepy.editor import VideoClip, VideoFileClip
import subprocess
import json
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
args = parser.parse_args()

folder_path = args.folderPath
video_fps = 30
video_dest_width = 2160
video_dest_height = 3840
wipe_left_time = 800

with open('./templates/template4/input/transition_span.json', 'r') as f:
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
    elif idx in (8, 9, 11):
        return intervals[idx - 1] + (transition_list[idx - 2].get("duration") + wipe_left_time) / 2000
    elif idx > 1 and idx < len(intervals):
        return intervals[idx - 1] + (transition_list[idx - 2].get("duration") + transition_list[idx - 1].get("duration")) / 2000
    return 0

def get_timestamp_without_transtion_span(idx):
    if idx == 1:
        return 0, intervals[0] - transition_list[0].get("duration") / 2000
    elif idx == len(intervals):
        return transition_list[-1].get("duration") / 1000, get_durations(idx)
    elif idx > 1 and idx < len(intervals):
        return transition_list[idx - 2].get("duration") / 1000, get_durations(idx) - transition_list[idx - 1].get("duration") / 1000
    return 0, 0

def write_videoclips(video_clip, idx):
    start, end = get_timestamp_without_transtion_span(idx)
    if idx in (11, 12):
        video_clip.subclip(0, video_clip.duration).write_videofile(
            f"{temp_folder}/{idx:02d}.mp4", fps=video_fps, temp_audiofile=f"{temp_folder}/{idx}.mp3", remove_temp=True)
    else:
        video_clip.subclip(0, video_clip.duration).write_videofile(
            f"{temp_folder}/{idx:02d}.mp4", fps=video_fps)

def create_moving_right_clip(image_path, idx):
    image = Image.open(image_path)
    image_width, image_height = image.size
    duration = get_durations(idx)
    def moving_right_frame(t):
        new_left = int((image_width - video_dest_width) * t / duration)
        cropped_image = image.crop((image_width - new_left - video_dest_width, 0, image_width - new_left, video_dest_height))
        return np.array(cropped_image)
    video_clip = VideoClip(moving_right_frame, duration=duration)
    write_videoclips(video_clip, idx)
    video_clip.close()

def create_resize_clip(video_path, idx):
    clip = VideoFileClip(video_path).subclip(0, get_durations(idx))
    aspect_ratio = clip.w / clip.h
    dest_ratio = video_dest_width / video_dest_height
    if aspect_ratio > dest_ratio:
        new_width = int(clip.h * dest_ratio)
        new_height = clip.h
    else:
        new_width = clip.w
        new_height = int(clip.w / dest_ratio)
    cropped_clip = clip.crop(x_center=(clip.w // 2), y_center=(clip.h // 2), width=new_width, height=new_height).resize((video_dest_width, video_dest_height))
    write_videoclips(cropped_clip, idx)
    clip.close()

def create_zoom_clip(image_path, idx, zoom_dest):
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    duration = get_durations(idx)
    def zoom_out_frame(t):
        zoom_factor = 1 - ((1 - zoom_dest) * (t / duration)**0.5)
        new_width = image_width * zoom_factor
        new_height = image_height * zoom_factor
        left = (image_width - new_width) / 2
        top = (image_height - new_height) / 2
        right = left + new_width
        bottom = top + new_height
        cropped_image = image.crop((left, top, right, bottom))
        video_aspect_ratio = video_dest_width / video_dest_height
        image_aspect_ratio = new_width / new_height
        if image_aspect_ratio > video_aspect_ratio:
            resize_height = video_dest_height
            resize_width = int(new_width * (video_dest_height / new_height))
        else:
            resize_width = video_dest_width
            resize_height = int(new_height * (video_dest_width / new_width))
        resized_image = cropped_image.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
        left = (resize_width - video_dest_width) / 2
        top = (resize_height - video_dest_height) / 2
        right = left + video_dest_width
        bottom = top + video_dest_height
        final_image = resized_image.crop((left, top, right, bottom))
        return np.array(final_image)
    video_clip = VideoClip(lambda t: zoom_out_frame(t), duration=duration)
    write_videoclips(video_clip, idx)
    video_clip.close()

def create_drop_clip(image_path, back_image, idx):
    fore_image = Image.open(image_path)
    image_width, image_height = fore_image.size
    duration = 1 / 4
    crop_height = int(image_width * video_dest_height / (video_dest_width * 2))
    image = fore_image.crop((0, image_height - crop_height, image_width, image_height)).resize((video_dest_width, video_dest_height // 2))
    def drop_from_top(t):
        if t >= duration:
            t = duration
        new_height = int(video_dest_height * t // (2 * duration))
        cropped_image = image.crop((0, video_dest_height // 2 - new_height, video_dest_width, video_dest_height // 2))
        back_image.paste(cropped_image, (0, 0))
        return np.array(back_image)
    video_clip = VideoClip(drop_from_top, duration=get_durations(idx))
    write_videoclips(video_clip, idx)
    video_clip.close()

def create_pop_clip(image_path, back_image, idx):
    fore_image = Image.open(image_path).convert("RGB")
    image_width, image_height = fore_image.size
    duration = 1 / 4
    crop_height = int(image_width * video_dest_height / (video_dest_width * 2))
    image = fore_image.crop((0, 0, image_width, crop_height)).resize((video_dest_width, video_dest_height // 2))
    def pop_from_bottom(t):
        if t >= duration:
            t = duration
        new_height = int(video_dest_height * t // (2 * duration))
        cropped_image = image.crop((0, 0, video_dest_width, new_height))
        back_image.paste(cropped_image, (0, video_dest_height - new_height))
        return np.array(back_image)
    video_clip = VideoClip(pop_from_bottom, duration=get_durations(idx))
    write_videoclips(video_clip, idx)
    video_clip.close()

def create_avatar_clip(video_path, idx):
    clip = VideoFileClip(video_path).subclip(0, get_durations(idx))
    aspect_ratio = clip.w / clip.h
    dest_ratio = video_dest_width / video_dest_height
    if aspect_ratio > dest_ratio:
        new_width = int(clip.size[1] * dest_ratio)
        new_height = clip.size[1]
    else:
        new_width = clip.size[0]
        new_height = int(clip.size[0] / dest_ratio)
    cropped_clip = clip.crop(x_center=(clip.w // 2), y_center=(clip.h // 2), width=new_width, height=new_height).resize((video_dest_width, video_dest_height))
    write_videoclips(cropped_clip, idx)
    clip.close()

def create_moving_and_zoom_clip(image_path, idx):
    def moving_and_zoom_frame(t):
        new_left = int(image_width / 4) - int((image_width - video_dest_width) * t / duration)

        zoom_factor = 1 - (1 - 1 / zoom_dest) * (duration - t) / duration

        new_width = int(video_dest_width * zoom_factor)
        new_height = int(video_dest_height * zoom_factor)
        
        left = (image_width - new_width) // 2 + new_left
        top = (image_height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        cropped_image = image.crop((left, top, right, bottom))
        resized_image = cropped_image.resize((video_dest_width, video_dest_height))
        
        return np.array(resized_image)

    rgba_image = Image.open(image_path)
    zoom_dest = 1.3
    duration = get_durations(idx)

    image = rgba_image.convert("RGB")
    image_width, image_height = image.size

    video_clip = VideoClip(moving_and_zoom_frame, duration=duration)
    write_videoclips(video_clip, idx)
    video_clip.close()

def create_resize_clip(video_path, idx):
    clip = VideoFileClip(video_path).subclip(0, get_durations(idx))
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
    write_videoclips(cropped_clip, idx)
    clip.close()

def create_blur_clip(video_path, idx):
    def blur_frame(frame):
        return cv2.GaussianBlur(frame, (51, 51), 10)
    clip = VideoFileClip(video_path).subclip(0, get_durations(idx))
    aspect_ratio = clip.w / clip.h
    dest_ratio = video_dest_width / video_dest_height
    if aspect_ratio > dest_ratio:
        new_width = int(clip.size[1] * dest_ratio)
        new_height = clip.size[1]
    else:
        new_width = clip.size[0]
        new_height = int(clip.size[0] / dest_ratio)
    cropped_clip = clip.crop(x_center=(clip.w // 2), y_center=(clip.h // 2), width=new_width, height=new_height).resize((video_dest_width, video_dest_height))
    background_clip = cropped_clip.fl_image(blur_frame)
    write_videoclips(background_clip, idx)
    clip.close()

def create_background_video():
    with ThreadPoolExecutor() as executor:
        futures = []
        futures.append(executor.submit(create_moving_right_clip, os.path.join(folder_path, "1.png"), 1))
        futures.append(executor.submit(create_moving_right_clip, os.path.join(folder_path, "2.png"), 2))
        futures.append(executor.submit(create_resize_clip ,os.path.join(folder_path, "ss.mp4"), 3))
        futures.append(executor.submit(create_moving_right_clip, os.path.join(folder_path, "3.png"), 4))
        futures.append(executor.submit(create_zoom_clip, os.path.join(folder_path, "4.png"), 5, 0.7))
        futures.append(executor.submit(create_drop_clip, os.path.join(folder_path, "5.png"), Image.open(os.path.join(folder_path, "4.png")), 6))
        futures.append(executor.submit(create_pop_clip, os.path.join(folder_path, "6.png"), Image.open(os.path.join(folder_path, "5.png")), 7))
        futures.append(executor.submit(create_avatar_clip, os.path.join(folder_path, "bg_avatar.mp4"), 8))
        futures.append(executor.submit(create_moving_and_zoom_clip, os.path.join(folder_path, "7.png"), 9))
        futures.append(executor.submit(create_resize_clip, os.path.join(folder_path, "ss.mp4"), 10))
        futures.append(executor.submit(create_avatar_clip, os.path.join(folder_path, "bg_avatar.mp4"), 11))
        futures.append(executor.submit(create_blur_clip, os.path.join(folder_path, "ss.mp4"), 12))
        
        for future in futures:
            future.result()

create_background_video()

# ====================== ADD TRANSITION BETWEEN THEM===========================
video_clip_names = [f"{i+1:02d}.mp4" for i in range(7)]
video_clip_names.append("08-09-10.mp4")
video_clip_names.append("11-12.mp4")

#add temp_folder to the video_clip_names
video_clip_names = [os.path.join(temp_folder, video_clip_name) for video_clip_name in video_clip_names]

background_video = os.path.join(temp_folder, "background_video.mp4")

command = ['ffmpeg-concat', '-T', "./templates/template1/input/transition.json",
           '-o', background_video, '-c 9'] + video_clip_names
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

for file_name in os.listdir(temp_folder):
    file_path = os.path.join(temp_folder, file_name)
if file_name != "background_video.mp4":
    os.remove(file_path)