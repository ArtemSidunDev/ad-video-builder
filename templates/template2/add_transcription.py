
import numpy as np
import json
from moviepy.editor import TextClip, CompositeVideoClip, AudioFileClip, CompositeAudioClip, VideoFileClip, ColorClip, ImageClip
import moviepy.audio.fx.all as afx
import os
import shutil
import math
import argparse
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser(description="Generate a background video.")
parser.add_argument('folderPath', type=str, help='Path to the folder')
parser.add_argument('subtitleSettings', type=str, help='Path to the settings file')
args = parser.parse_args()

folder_path = args.folderPath
subtitle_settings_path = args.subtitleSettings

try:
    with open(subtitle_settings_path, 'r') as f:
        subtitle_settings = json.load(f)
    print("Parsed JSON:", subtitle_settings)
except Exception as e:
    print("Failed to read or parse JSON:", e)

FONT_SIZE = 64

FONT = f"./templates/common/input/{subtitle_settings.get('font', 'ProximaNova-Black')}.ttf"
FONT_COLOR = subtitle_settings.get('fontColor', '#FFFFFF')
FONT_OUTLINE_COLOR = subtitle_settings.get('fontOutlineColor', '#000000')
FONT_HIGHLIGHT_COLOR = subtitle_settings.get('fontHighlightColor', '#D0AA3A')


FONT_OUTLINE_WIDTH = 4
HIGHLIGHT_RADIUS = 21
FONT_MARGIN = 20

video_dest_width = 1216
video_dest_height = 2160

foreground_audio = VideoFileClip(os.path.join(folder_path, "avatar.mp4")).audio

temp_folder = os.path.join(folder_path, "temp")
if not os.path.exists(temp_folder):
    os.makedirs(temp_folder)
    

with open(os.path.join(folder_path, 'transcription.json'), 'r') as f:
    wordlevel_info = json.load(f)

def get_sentences_from_words(word_list):
    # Initialize an empty list to store sentences
    sentences = []
    # Initialize an empty string to store the current sentence
    current_sentence = ""
    # Initialize variables to track the start and end indices of each sentence
    start_idx = 0
    end_idx = 0
    
    # Iterate through each word in the word list
    for word_info in word_list:
        # Increment the end index
        end_idx += 1
        
        # Append the word to the current sentence
        current_sentence += word_info['word'] + " "
        
        # Check if the word ends with a punctuation mark denoting the end of a sentence
        if word_info['word'][-1] in ['.', '!', '?',',']:
            # Add the current sentence to the list of sentences
            sentences.append((current_sentence.strip(), start_idx, end_idx - 1))
            # Reset the current sentence and update the start index
            current_sentence = ""
            start_idx = end_idx
    
    # Add the last sentence if it's not empty
    if current_sentence:
        sentences.append((current_sentence.strip(), start_idx, end_idx))
    return sentences

def split_text_into_lines(word_list):
    subtitles = []

    sentences = get_sentences_from_words(word_list)
    
    for data in sentences:
        line = []
        line_duration = 0
        
        MaxChars = 30
        MaxDuration = 2.0
        MaxGap = 1.5
        
        sentence, start_idx, end_idx = data
        count = len( sentence)
        
        MaxChars = math.ceil( count/(math.ceil(count/MaxChars)))
        
        for idx in range( start_idx, end_idx+1):
            if idx >= len(word_list):
                break
            word_data = word_list[idx]
            start = word_data["start"]
            end = word_data["end"]
    
            line.append(word_data)
            line_duration += end - start
    
            temp = " ".join(item["word"] for item in line)
    
            # Check if adding a new word exceeds the maximum character count or
            # duration
            new_line_chars = len(temp)
    
            duration_exceeded = line_duration > MaxDuration
            chars_exceeded = new_line_chars > MaxChars
            if idx > 0:
                gap = word_data['start'] - word_list[idx - 1]['end']
                # print (word,start,end,gap)
                maxgap_exceeded = gap > MaxGap
            else:
                maxgap_exceeded = False
    
            if duration_exceeded or chars_exceeded or maxgap_exceeded:
                if line:
                    subtitle_line = {
                        "word": " ".join(item["word"] for item in line),
                        "start": line[0]["start"],
                        "end": line[-1]["end"],
                        "textcontents": line
                    }
                    subtitles.append(subtitle_line)
                    line = []
                    line_duration = 0
        if line:
            subtitle_line = {
                "word": " ".join(item["word"] for item in line),
                "start": line[0]["start"],
                "end": line[-1]["end"],
                "textcontents": line
            }
            subtitles.append(subtitle_line)
    return subtitles


linelevel_subtitles = split_text_into_lines(wordlevel_info)

for line in linelevel_subtitles:
    json_str = json.dumps(line, indent=4)

def is_in_range(number, ranges):
    for start, end in ranges:
        if start <= number <= end:
            return True
    return False

def generate_outline_offsets(radius):
    offsets = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if dx**2 + dy**2 <= radius**2:
                offsets.append((dx, dy))
    return offsets

def generate_outline_text(
    text,
    font,
    fontsize,
    color,
    outline_color,
    pos,
    start,
    duration,
    stroke_width=FONT_OUTLINE_WIDTH,
    method="caption"
):
    clips = []

    dx_dy = [
        (-3, 0), (3, 0), (0, -3), (0, 3),
        (-2, -2), (-2, 2), (2, -2), (2, 2),
        (-3, -2), (-3, 2), (3, -2), (3, 2),
        (-2, -3), (-2, 3), (2, -3), (2, 3),
        (-3, -3), (-3, 3), (3, -3), (3, 3)
    ]

    for dx, dy in dx_dy:
        shadow = TextClip(
            text,
            font=font,
            fontsize=fontsize,
            color=outline_color,
            method=method
        ).set_start(start).set_duration(duration).set_position((pos[0] + dx, pos[1] + dy))
        clips.append(shadow)

    fill = TextClip(
        text,
        font=font,
        fontsize=fontsize,
        color=color,
        method=method
    ).set_start(start).set_duration(duration).set_position(pos)
    clips.append(fill)

    return clips

def generate_highlight_text(
    word,
    font,
    fontsize,
    color,
    highlightcolor,
    outline_color,
    position,
    duration,
    radius=HIGHLIGHT_RADIUS,
    margin=FONT_MARGIN,
    method="caption"
):
    clips = []
    dx_dy = [
        (-3, 0), (3, 0), (0, -3), (0, 3),
        (-2, -2), (-2, 2), (2, -2), (2, 2),
        (-3, -2), (-3, 2), (3, -2), (3, 2),
        (-2, -3), (-2, 3), (2, -3), (2, 3),
        (-3, -3), (-3, 3), (3, -3), (3, 3)
    ]

    temp_clip = TextClip(word, font=font, fontsize=fontsize, color=color, method=method)
    w, h = temp_clip.size
    bg_size = (w + margin, h)

    mask_image = Image.new("RGB", bg_size, 0)
    draw = ImageDraw.Draw(mask_image)
    draw.rounded_rectangle([(0, 0), bg_size], radius=radius, fill=255)
    mask_clip = ImageClip(np.array(mask_image), ismask=True).set_duration(duration)

    for dx, dy in dx_dy:
        shadow = TextClip(
            word,
            font=font,
            fontsize=fontsize,
            color=color,
            bg_color=outline_color,
            size=bg_size,
            method=method
        ).set_mask(mask_clip).set_duration(duration).set_position((position[0] + dx, position[1] + dy))
        clips.append(shadow)

    main = TextClip(
        word,
        font=font,
        fontsize=fontsize,
        color=color,
        bg_color=highlightcolor,
        size=bg_size,
        method=method
    ).set_mask(mask_clip).set_duration(duration).set_position(position)
    clips.append(main)

    return clips

def create_caption(
        textJSON,
        framesize,
        font=FONT,
        fontsize=FONT_SIZE,
        color=FONT_COLOR,
        highlightcolor=FONT_HIGHLIGHT_COLOR):

    full_duration = textJSON['end'] - textJSON['start']
    word_clips = []
    xy_textclips_positions = []

    x_pos = 0
    y_pos = 0

    frame_width, frame_height = framesize
    x_buffer = frame_width * 1 / 10

    # Variables to track the width and height of a space
    space_clip = TextClip(" ", fontsize=fontsize, color=color)
    space_width = space_clip.size[0] - 24
    space_height = 0
    
    # Variables to track the current position
    x_pos, y_pos = 0, 0
    line_widths = []
    line_heights = []
    
    current_line_width = 0

     # First pass: calculate the width and height of each word and line
    for wordJSON in textJSON['textcontents']:
        word_clip = TextClip(wordJSON['word'], font=font, fontsize=fontsize, color=color, stroke_color=FONT_OUTLINE_COLOR, stroke_width=FONT_OUTLINE_WIDTH)
        word_width, word_height = word_clip.size

        if x_pos + word_width + space_width > frame_width - 2 * x_buffer:
            line_heights.append(y_pos + word_height)  # Store the height of the line
            line_widths.append(current_line_width) # Store the width of the line
            x_pos, y_pos = 0, y_pos + word_height + space_height  # Move to the next line
            current_line_width = 0 # Reset the current line width

        xy_textclips_positions.append({
            "x_pos": x_pos,
            "y_pos": y_pos,
            "width": word_width,
            "height": word_height,
            "word": wordJSON['word'],
            "start": wordJSON['start'],
            "end": wordJSON['end'],
            "duration": wordJSON['end'] - wordJSON['start']
        })

        x_pos += word_width + space_width
        current_line_width += word_width + space_width

    # Add the last line height
    line_heights.append( word_height)
    line_widths.append( current_line_width)
    # Calculate the total height of all lines and find the starting y position
    total_text_height = sum(line_heights)
    # Calculate the starting y position should be 43% of the frame height
    start_y_pos = frame_height * 0.43 

    # Second pass: set the position of each word clip
    current_line = 0
    for word_info in xy_textclips_positions:
        
        current_line = word_info['y_pos'] // word_height # Move to the next line

        # Center the line horizontally
        centered_x_pos = (frame_width - line_widths[current_line]) / 2 + word_info['x_pos']
        move_text_up_time_ranges = []

        if is_in_range(textJSON['start'], move_text_up_time_ranges):
            start_y_pos = frame_height * 0.30
        
        y_offset = 0
        if textJSON['start'] >= 13  and textJSON['end'] <= 16:
            y_offset = 100
        elif textJSON['start'] >= 25:
            y_offset = -450

        # print("y_offset", textJSON['start'], textJSON['end'], y_offset, textJSON['word'])
        
        word_clip = TextClip(word_info['word'], font=font, fontsize=fontsize, color=color, stroke_color=FONT_OUTLINE_COLOR, stroke_width=FONT_OUTLINE_WIDTH).set_start( textJSON['start']).set_duration( full_duration)
        word_clip = word_clip.set_position((centered_x_pos, start_y_pos + y_offset + word_info['y_pos'] * 0.8))
        word_clips.append(word_clip)
        
        word_clip_temp = TextClip(word_info['word'], font=font, fontsize=fontsize, color=color, stroke_color=FONT_OUTLINE_COLOR, stroke_width=FONT_OUTLINE_WIDTH, bg_color=FONT_HIGHLIGHT_COLOR)
        word_clip_highlight = TextClip(word_info['word'], font=font, fontsize=fontsize, color=color, stroke_color=FONT_OUTLINE_COLOR, stroke_width=FONT_OUTLINE_WIDTH, bg_color=FONT_HIGHLIGHT_COLOR, size=(word_clip_temp.w+FONT_MARGIN,word_clip_temp.h), method="caption")
        mask_image = Image.new("RGB", (word_clip_highlight.w, word_clip_highlight.h), 0)
        draw = ImageDraw.Draw(mask_image)
        draw.rounded_rectangle(
            [(0, 0), (word_clip_highlight.w, word_clip_highlight.h)],
            radius=HIGHLIGHT_RADIUS,
            fill=255
        )
        mask_clip = ImageClip(np.array(mask_image), ismask=True).set_duration(word_info['duration'])
        word_clip_highlight = word_clip_highlight.set_mask(mask_clip).set_start(word_info['start']).set_duration(word_info['duration'])
        word_clip_highlight = word_clip_highlight.set_position( ( centered_x_pos-FONT_MARGIN//2, start_y_pos + y_offset +  word_info['y_pos'] * 0.8))       
        word_clips.append(word_clip_highlight)

        outline_clips = generate_outline_text(
            text=word_info['word'],
            font=font,
            fontsize=fontsize,
            color=color,
            outline_color=FONT_OUTLINE_COLOR,
            pos=(centered_x_pos, start_y_pos + y_offset + word_info['y_pos'] * 0.8),
            start=textJSON['start'],
            duration=full_duration
        )
        word_clips.extend(outline_clips)

    return word_clips

frame_size = (video_dest_width, video_dest_height)

all_linelevel_splits = []

for line in linelevel_subtitles:
    out = create_caption(line, frame_size)
    all_linelevel_splits.extend(out)

# Load the input video
input_video = VideoFileClip(os.path.join(temp_folder, "background_video.mp4"))
# Get the duration of the input video
input_video_duration = input_video.duration

# If you want to overlay this on the original video uncomment this and
# also change frame_size, font size and color accordingly.
final_video = CompositeVideoClip([input_video] + all_linelevel_splits)

camera_audio = AudioFileClip("./templates/template2/input/camera.mp3").subclip(0.4, 0.8).fx(afx.audio_normalize).fx(afx.volumex, 0.5)
transition_audio = AudioFileClip("./templates/template2/input/transition.wav").subclip(0, 0.4).fx(afx.audio_normalize).fx(afx.volumex, 0.3)

audios = [foreground_audio.fx( afx.audio_normalize)]

avatar_video = VideoFileClip(os.path.join(folder_path, "avatar.mp4"))

audios.append( AudioFileClip(os.path.join(folder_path, "background_audio.mp3")).fx(
        afx.audio_normalize).fx(
            afx.volumex,
        0.1).set_duration(input_video_duration))

audios.append(camera_audio.set_start(1.8))
audios.append(transition_audio.set_start(3.8))
audios.append(transition_audio.set_start(6.8))
audios.append(camera_audio.set_start(8.8))
audios.append(camera_audio.set_start(10.8))
audios.append(transition_audio.set_start(13.8))
audios.append(transition_audio.set_start(15.8))
audios.append(camera_audio.set_start(18.3))
audios.append(camera_audio.set_start(20.8))
audios.append(transition_audio.set_start(25.6))
audios.append(transition_audio.set_start(avatar_video.duration - 0.5))

audios.append(
    VideoFileClip(os.path.join(folder_path, "action.mp4")).audio.fx(
        afx.audio_normalize).fx( afx.volumex, 0.8).set_start(avatar_video.duration + 1))

composite_audio_clip = CompositeAudioClip(audios)
final_video = final_video.set_audio(composite_audio_clip)

# Save the final clip as a video file with the audio included
final_video.subclip(0, input_video_duration).write_videofile(os.path.join(folder_path, "output.mp4"), temp_audiofile=f"{temp_folder}/output.aac", remove_temp=True, codec="libx264", audio_codec="aac", fps=30)

try:
    shutil.rmtree(temp_folder)
    print(f"Folder '{temp_folder}' deleted successfully.")
except OSError as e:
    print(f"Error: {e}")