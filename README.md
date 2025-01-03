# Face Detection and Tracking in Video using RetinaFace and Face_Recognition

This repository provides a Python script to detect and track a specific individual's face in a video. The pipeline combines the **RetinaFace-pytorch** model for face detection and the **face_recognition** library for face identification based on a reference image. The script minimizes jitter and ID flip during tracking to ensure smooth face detection and consistent tracking.

---

## Installation

### Prerequisites
- **Python 3.8 or later**
- **GPU support with CUDA installed**

### Install Dependencies

1. **Install the RetinaFace library**:
   ```bash
   pip install retinaface-pytorch
   Reference: RetinaFace-Pytorch Documentation
   
2. **Install the face recognition library:**:
   ```bash
    pip install face_recognition
    Reference: Face_Recognition GitHub
   
3. **Install other required libraries::**:
   ```bash
    pip install opencv-python numpy tqdm pillow pandas scipy

   


