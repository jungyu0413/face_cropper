# Face Detection and Tracking in Video using RetinaFace and Face_Recognition

This repository provides a Python script to detect and track a specific individual's face in a video. The pipeline combines the **RetinaFace-pytorch** model for face detection and the **face_recognition** library for face identification based on a reference image. The script minimizes jitter and ID flip during tracking to ensure smooth face detection and consistent tracking.

---

## Installation

### Prerequisites
- **Python 3.8 or later**
- **GPU support with CUDA installed**

### Install Dependencies

1. **Install the face detection:**
   ```bash
   git clone https://github.com/biubug6/Pytorch_Retinaface.git
   Pytorch version 1.1.0+ and torchvision 0.3.0+ are needed.
   Codes are based on Python 3
   
2. **Install the face recognition library:**
   ```bash
    pip install face_recognition
    Reference: Face_Recognition GitHub
   
3. **Install other required libraries:**
   ```bash
    pip install opencv-python numpy tqdm pillow pandas scipy

   
4. **face recognition and detection:**
If a reference image exists, specify the path in sample_img_path = ''. The script will parse and crop the face with the highest similarity to the reference image.

   ```bash
    python face_crop.py

6. **video filter:**
Provide a video file and a frame-wise annotation file, then specify the labels you want to parse. The script will filter and save clips that correspond to the specified labels.
   ```bash
    python face_crop.py
