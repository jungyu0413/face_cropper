import numpy as np
from scipy.spatial.distance import euclidean, cosine
from PIL import Image
import cv2
from tqdm import tqdm

def resize_image(img, max_width=640, max_height=480):
    """
    이미지 크기를 조정합니다.
    - 만약 이미지의 가로/세로 크기가 설정된 최대 크기를 초과하면 비율을 유지하며 리사이징합니다.
    - 그렇지 않으면 원본 이미지를 반환합니다.
    """
    height, width = img.shape[:2]
    if width > max_width or height > max_height:
        scale = min(max_width / width, max_height / height)
        img = cv2.resize(img, (int(width * scale), int(height * scale)))
    return img


def get_most_similar_face(current_faces, reference_landmarks):
    """
    주어진 얼굴 목록에서 이전 프레임의 랜드마크와 가장 유사한 얼굴을 선택합니다.
    - 랜드마크 좌표 간의 평균 유클리드 거리를 계산하여 유사도를 평가합니다.
    """
    min_distance = float('inf')
    best_face = None

    for face in current_faces:
        landmarks = face['landmarks']

        # 각 랜드마크 (왼쪽 눈, 오른쪽 눈, 코)에 대한 유클리드 거리 계산
        distances = [
            euclidean(landmarks[0], reference_landmarks[0]),  # 왼쪽 눈
            euclidean(landmarks[1], reference_landmarks[1]),  # 오른쪽 눈
            euclidean(landmarks[2], reference_landmarks[2]),  # 코
        ]
        current_distance = np.mean(distances)  # 평균 거리 계산

        if current_distance < min_distance:
            min_distance = current_distance
            best_face = face

    return best_face


def make_square_box(box, img_shape):
    """
    주어진 사각형 박스를 정사각형으로 조정합니다.
    - 중심을 기준으로 정사각형을 만듭니다.
    - 이미지 경계를 초과하지 않도록 조정합니다.
    """
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    side = max(width, height)

    # 중심 좌표 계산
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # 새로운 정사각형 좌표 계산
    x1_new = max(0, cx - side // 2)
    y1_new = max(0, cy - side // 2)
    x2_new = min(img_shape[1], cx + side // 2)
    y2_new = min(img_shape[0], cy + side // 2)

    return [x1_new, y1_new, x2_new, y2_new]


def select_best_face_based_on_embedding(face_recognition, faces, embedding, img, img_shape=None, score_weight=0.3, size_weight=0.2, embedding_weight=0.5):
    """
    임베딩 유사도, 신뢰도 점수, 박스 크기를 기반으로 가장 적합한 얼굴을 선택합니다.
    - 얼굴별로 임베딩을 계산하고 주어진 임베딩과 코사인 유사도를 평가합니다.
    - 가중치를 사용해 각 요소를 종합 점수로 계산합니다.
    """
    best_face = None
    best_score = -1

    for face in faces:
        # 얼굴 영역 추출
        box = face['bbox']
        x1, y1, x2, y2 = map(int, box)

        # 얼굴 이미지를 잘라내고 PIL 이미지로 변환
        cropped_face = img[y1:y2, x1:x2]
        pil_face = Image.fromarray(cropped_face).convert('RGB')

        # 현재 얼굴의 임베딩 계산
        try:
            face_embedding = face_recognition.face_encodings(np.array(pil_face))[0]
        except IndexError:
            # 임베딩 계산 불가능한 경우 건너뜀
            continue

        # 신뢰도 점수
        confidence = face['score']

        # 박스 크기 계산 (이미지 전체 크기로 정규화)
        box_area = (x2 - x1) * (y2 - y1)
        normalized_box_area = box_area / (img_shape[0] * img_shape[1])

        # 코사인 유사도 계산
        embedding_similarity = 1 - cosine(embedding, face_embedding)

        # 가중치 점수 계산
        combined_score = (confidence * score_weight +
                          normalized_box_area * size_weight +
                          embedding_similarity * embedding_weight)

        # 최고 점수 갱신
        if combined_score > best_score:
            best_score = combined_score
            best_face = face

    return best_face



def detect_and_save_faces_with_landmark_tracking(video_path, output_path, embedding, face_detector, face_recognition):
    """
    비디오에서 특정 개체의 얼굴을 탐지하고 저장합니다.
    - 첫 번째 프레임에서는 임베딩 기반 또는 가장 큰 박스와 높은 신뢰도 기반으로 얼굴을 선택합니다.
    - 이후 프레임에서는 이전 프레임의 랜드마크와의 유사성을 기준으로 얼굴을 선택합니다.
    """
    cap = cv2.VideoCapture(video_path)

    previous_landmarks = None
    first_frame = True
    frame_width, frame_height = None, None

    # 비디오 총 프레임 수 계산
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

            # 첫 번째 프레임에서만 리사이즈 적용
            if first_frame and (img.shape[1] > 1920 or img.shape[0] > 1080):
                img = resize_image(img)

            # 비디오 저장 초기화
            if frame_width is None or frame_height is None:
                frame_height, frame_width = img.shape[:2]
                writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # 얼굴 탐지
            annotations = face_detector.predict_jsons(img)

            if not annotations:
                print("No faces detected. Skipping frame.")
                pbar.update(1)
                continue

            if first_frame:
                if not annotations:
                    print("No faces detected in the first frame. Skipping frame.")
                    pbar.update(1)
                    continue
                if embedding is not False:
                    # 임베딩 기반으로 얼굴 선택
                    best_face = select_best_face_based_on_embedding(face_recognition, annotations, embedding, img, img_shape=img.shape)
                else:
                    # 신뢰도가 높고 크기가 가장 큰 얼굴 선택
                    try:
                        best_face = max(
                            annotations,
                            key=lambda face: (face['score'], (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
                        )
                    except ValueError:
                        print("No valid face detected in the first frame. Skipping frame.")
                        pbar.update(1)
                        continue
                first_frame = False
            else:
                # 이후 프레임 처리
                if not annotations:
                    print("No faces detected. Skipping frame.")
                    pbar.update(1)
                    continue
                try:
                    best_face = get_most_similar_face(annotations, previous_landmarks)
                except IndexError:
                    print("No valid face found for comparison. Skipping frame.")
                    pbar.update(1)
                    continue

            # 이후 로직
            if best_face is None:
                print("No matching face found. Skipping frame.")
                pbar.update(1)
                continue

            # 랜드마크 업데이트
            previous_landmarks = best_face['landmarks']

            # 얼굴 잘라내기
            box = best_face['bbox']
            x1, y1, x2, y2 = map(int, box)
            cropped_face = img[y1:y2, x1:x2]

            # 정사각형 박스 조정 및 리사이즈
            cropped_face = make_square_box([x1, y1, x2, y2], img.shape)
            x1, y1, x2, y2 = cropped_face
            cropped_face = img[y1:y2, x1:x2]
            resized_face = cv2.resize(cropped_face, (frame_width, frame_height))
            writer.write(resized_face)

            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Video saved at {output_path}")
    return True
