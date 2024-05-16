
# Project Setup Instructions

This README outlines the steps to install and set up the necessary environment for the project.

## Prerequisites

Ensure you have administrative access (sudo privileges) on your system to perform these installations.

## Installation Steps

### 1. Install Python and Node.js

Install Python 3.11.5 and Node.js 20.11.0:

### 2. Install FFmpeg
Install FFmpeg version greater than 4.3.0:
```
sudo apt install ffmpeg
```
Verify the installation and check the version:
```
ffmpeg -version
```
### 3. Install Python Dependencies
Install the required Python packages using pip:
```
pip install -r requirements.txt
```
### 4. Install ffmpeg-concat Globally
Install ffmpeg-concat using npm:
```
npm install -g ffmpeg-concat
```

### 5. Install whisper-timestamped
https://github.com/linto-ai/whisper-timestamped
Install whisper-timestamped using pip3:
```
pip3 install whisper-timestamped
```

### 6. Install project dependency
Install whisper-timestamped using pip3:
```
npm install
```