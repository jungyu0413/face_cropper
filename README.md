<!DOCTYPE html>
<html>
<head>
    <title>Face Cropper</title>
</head>
<body>
    <h1>Face Cropper</h1>
    <p>
        The <strong>Face Cropper</strong> tool processes video files to detect faces in each frame and crops them into square regions. The tool uses the <a href="https://github.com/serengil/retinaface">RetinaFace</a> library for face detection and ensures consistent tracking of the same individual across frames.
    </p>

    <h2>Features</h2>
    <ul>
        <li>Processes video frame by frame.</li>
        <li>Detects and crops faces into square bounding boxes.</li>
        <li>Tracks the closest face across frames.</li>
        <li>Exports cropped faces into a new video file.</li>
    </ul>

    <h2>Installation</h2>
    <pre>
<code>pip install retina-face</code>
    </pre>

    <h2>Usage</h2>
    <pre>
<code>import cv2
from your_module import detect_and_save_faces_with_visualization

video_path = "path_to_your_video.mp4"
output_path = "output_video.mp4"

detect_and_save_faces_with_visualization(video_path, output_path)</code>
    </pre>

    <h2>Example Output</h2>
    <p>The tool will save a new video where each frame contains the cropped face detected in the input video.</p>

    <h2>Notes</h2>
    <ul>
        <li>Ensure the input video is in a compatible format (e.g., MP4).</li>
        <li>Make sure to install all required dependencies.</li>
    </ul>
</body>
</html>
