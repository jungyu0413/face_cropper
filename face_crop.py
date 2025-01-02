import cv2
import numpy as np
import pandas as pd
from retinaface import RetinaFace
from scipy.spatial.distance import euclidean
from tqdm import tqdm

def resize_image(img, max_width=640, max_height=480):
    """
    Resize image if it exceeds the maximum dimensions.
    """
    height, width = img.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return img

def get_closest_box(current_boxes, previous_box, max_distance=50):
    """
    Find the closest box to the previous box based on the center point distance.
    """
    previous_center = np.array([
        (previous_box[0] + previous_box[2]) / 2,
        (previous_box[1] + previous_box[3]) / 2
    ])
    min_distance = float('inf')
    best_box = None

    for box in current_boxes:
        current_center = np.array([
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2
        ])
        distance = euclidean(previous_center, current_center)
        if distance < min_distance:
            min_distance = distance
            best_box = box

    # If the closest box is too far, return None
    if min_distance > max_distance:
        return None

    return best_box

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
def detect_and_save_faces_optimized(video_path, output_path, max_distance=50):
    cap = cv2.VideoCapture(video_path)
    previous_box = None
    frame_width, frame_height = None, None

    # 총 프레임 수 계산
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # MP4 저장 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    # tqdm으로 진행률 표시
    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            # 프레임 크기 초기화
            if frame_width is None or frame_height is None:
                frame_height, frame_width = img.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # 얼굴 검출
            resp = RetinaFace.detect_faces(img)

            if not resp:
                print("No faces detected.")
                if previous_box is not None:
                    x1, y1, x2, y2 = previous_box
                    cropped_face = img[y1:y2, x1:x2]
                    resized_face = cv2.resize(cropped_face, (frame_width, frame_height))
                    writer.write(resized_face)
                pbar.update(1)
                continue

            # 가장 높은 확신도의 박스 또는 이전 박스와 가까운 박스 선택
            boxes = [face_data['facial_area'] for face_data in resp.values()]
            scores = [face_data['score'] for face_data in resp.values()]

            if previous_box is None:
                best_index = np.argmax(scores)
                best_box = boxes[best_index]
            else:
                best_box = get_closest_box(boxes, previous_box, max_distance=max_distance)
                if best_box is None:
                    best_box = previous_box

            # 정사각형으로 박스 조정
            best_box = make_square_box(best_box, img.shape)
            previous_box = best_box

            # 얼굴 자르기
            x1, y1, x2, y2 = best_box
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
        detect_and_save_faces_optimized(vid_path, save_path)

if __name__ == '__main__':
    main()