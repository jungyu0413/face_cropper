import cv2
import numpy as np
from retinaface import RetinaFace
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import pandas as pd

def resize_image(img, max_width=640, max_height=480):
    """
    Resize image if it exceeds the maximum dimensions.
    """
    height, width = img.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return img

def get_most_similar_face(current_faces, reference_landmarks):
    """
    Find the face with the most similar landmarks to the reference landmarks.
    """
    min_distance = float('inf')
    best_face = None

    for face in current_faces:
        landmarks = face['landmarks']

        # Calculate distances for each key point
        distances = [
            euclidean(landmarks['left_eye'], reference_landmarks['left_eye']),
            euclidean(landmarks['right_eye'], reference_landmarks['right_eye']),
            euclidean(landmarks['nose'], reference_landmarks['nose']),
        ]
        current_distance = np.mean(distances)  # Use the mean distance for similarity

        if current_distance < min_distance:
            min_distance = current_distance
            best_face = face

    return best_face

def make_square_box(box, img_shape):
    """
    Adjust the box to make it square, keeping the center and staying within image boundaries.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    side = max(width, height)

    # Center coordinates
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Calculate new square coordinates
    x1_new = max(0, cx - side // 2)
    y1_new = max(0, cy - side // 2)
    x2_new = min(img_shape[1], cx + side // 2)
    y2_new = min(img_shape[0], cy + side // 2)

    return [x1_new, y1_new, x2_new, y2_new]

def detect_and_save_faces_with_landmark_tracking(video_path, output_path):
    """
    Detect and save the face of a specific individual across a video.
    """
    cap = cv2.VideoCapture(video_path)
    previous_landmarks = None
    frame_width, frame_height = None, None

    # 총 프레임 수 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # MP4 저장 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            # Resize image if necessary
            img = resize_image(img)

            # Initialize frame dimensions and video writer
            if frame_width is None or frame_height is None:
                frame_height, frame_width = img.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # 얼굴 검출
            resp = RetinaFace.detect_faces(img)

            if not resp:
                print("No faces detected.")
                pbar.update(1)
                continue

            faces = list(resp.values())
            if previous_landmarks is None:
                # 첫 프레임에서는 가장 높은 확신도의 얼굴 선택
                best_face = max(faces, key=lambda x: x['score'])
            else:
                # 이전 프레임의 얼굴과 가장 유사한 얼굴 선택
                best_face = get_most_similar_face(faces, previous_landmarks)

            if best_face is None:
                print("No matching face found.")
                pbar.update(1)
                continue

            # 랜드마크 업데이트
            previous_landmarks = best_face['landmarks']

            # 얼굴 박스 크롭
            box = best_face['facial_area']
            x1, y1, x2, y2 = map(int, box)
            cropped_face = img[y1:y2, x1:x2]

            # 박스 정사각형으로 조정
            cropped_face = make_square_box([x1, y1, x2, y2], img.shape)

            # 얼굴 리사이즈 후 저장
            x1, y1, x2, y2 = cropped_face
            cropped_face = img[y1:y2, x1:x2]
            resized_face = cv2.resize(cropped_face, (frame_width, frame_height))
            writer.write(resized_face)

            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Video saved at {output_path}")


def main():
    video_df_path = ' '
    video_df = pd.read_csv(video_df)  # Load video paths
    for dt in tqdm(video_df.iterrows()):
        vid_path = dt[1].path
        save_path = dt[1].path.replace('orgin_data', 'face_data')
        detect_and_save_faces_with_landmark_tracking(vid_path, save_path)

if __name__ == '__main__':
    main()