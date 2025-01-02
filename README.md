# Face Cropper Tool

This tool processes videos, detects faces frame by frame using the RetinaFace library, and saves cropped face regions as a new video.

## Installation

First, install the required dependencies:

```bash
pip install retina-face opencv-python tqdm scipy matplotlib

from your_script_name import detect_and_save_faces

video_path = "path/to/your/video.mp4"
output_path = "path/to/save/cropped_faces.mp4"

detect_and_save_faces(video_path, output_path)
