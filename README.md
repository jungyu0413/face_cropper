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





### Issue report

#### 1. ID flip : 영상 내의 대상자가 다수가 등장(초반 또는 중반)하는 경우
      1) threshold를 기준으로 초반 프레임에서 등장하는 다수의 얼굴을 인식 및 임베딩 저장 이후 트래킹하는 방식.
      2) 일정하게 지속적으로 등장하는 사람의 얼굴만 크롭 및 저장.
#### 2. Black frame : 초반의 해상도, 밝기가 0에 수렴하는 경우
      1) 1번의 방식으로 해결 가능.
#### 3. Non Frontalization : 얼굴이 옆면 또는 뒷면을 보는 경우
      1) 옆면 임베딩과 뒷면 임베딩의 차이를 확인.
#### 4. Non Same size box : 얼굴을 크롭하는 박스의 사이즈가 다른 경우
      1) 전체 프레임에서 코를 중점으로 잡아서 눈 입의 위치를 일정한 비율로 잡도록 설정.
