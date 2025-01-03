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



def detect_and_save_faces_with_landmark_tracking(video_path, output_path, embedding, face_detector, cfg, face_recognition, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    previous_landmarks = None
    previous_box = None
    first_frame = True
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            if first_frame and (img.shape[1] > 1920 or img.shape[0] > 1080):
                img = resize_image(img)

            # RetinaFace 추론
            boxes, scores, landms = retinaface_inference(face_detector, img, cfg, device)

            top_k = 20
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]
            
            if first_frame:
                if len(boxes) == 0:
                    print("No faces detected in the first frame. Skipping frame.")
                    pbar.update(1)
                    continue

                if embedding is not False:
                    # 임베딩 기반으로 얼굴 선택
                    annotations = [{'bbox': box, 'score': score, 'landmarks': landm} for box, score, landm in zip(boxes, scores, landms)]
                    best_face = select_best_face_based_on_embedding(face_recognition, annotations, embedding, img, img_shape=img.shape)
                else:
                    # 신뢰도와 박스 크기에 기반한 선택
                    annotations = [{'bbox': box, 'score': score, 'landmarks': landm} for box, score, landm in zip(boxes, scores, landms)]
                    best_face = max(annotations, key=lambda face: (face['score'], (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1])))

                first_frame = False
            else:
                if len(boxes) == 0:
                    print("No faces detected. Skipping frame.")
                    pbar.update(1)
                    continue

                annotations = [{'bbox': box, 'score': score, 'landmarks': landm} for box, score, landm in zip(boxes, scores, landms)]
                best_face = get_most_similar_face(annotations, previous_landmarks)

            if best_face is None:
                print("No matching face found. Skipping frame.")
                pbar.update(1)
                continue

            # 랜드마크 및 박스 업데이트
            previous_landmarks = best_face['landmarks']
            box = calculate_box_from_landmarks(previous_landmarks, img.shape, previous_box)
            previous_box = box

            # 크롭된 얼굴 추출
            x1, y1, x2, y2 = map(int, box)
            cropped_face = img[y1:y2, x1:x2]

            # 전체 이미지의 평균 픽셀 값으로 패딩 영역 채우기
            avg_pixel_value = img.mean(axis=(0, 1), dtype=np.uint8)  # 평균 픽셀 값 계산
            padded_frame = np.full((frame_height, frame_width, 3), avg_pixel_value, dtype=np.uint8)

            # 크롭된 얼굴 이미지를 중앙에 삽입
            crop_height, crop_width = cropped_face.shape[:2]
            start_y = (frame_height - crop_height) // 2
            start_x = (frame_width - crop_width) // 2
            padded_frame[start_y:start_y + crop_height, start_x:start_x + crop_width] = cropped_face

            # 프레임 저장
            writer.write(padded_frame)
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Video saved at {output_path}")
    return True




import numpy as np
from scipy.spatial.distance import euclidean, cosine
from PIL import Image
import cv2
from tqdm import tqdm
import pandas as pd
import torch
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

# RetinaFace 모델 로드 함수
def load_retinaface_model(model, model_path, device='cuda'):
    cfg = {
        'name': 'Resnet50',
        'min_sizes': [[16, 32], [64, 128], [256, 512]],
        'steps': [8, 16, 32],
        'variance': [0.1, 0.2],
        'clip': False,
    }
    pretrained_dict = torch.load(model_path)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    model.eval()
    model.to(device)
    return model, cfg


# RetinaFace 추론 함수
def retinaface_inference(net, img_raw, cfg, device, top_k=20, nms_threshold=0.4):
    """
    RetinaFace 추론 수행
    - 입력 이미지를 기반으로 박스, 스코어, 랜드마크 추출
    - NMS 및 상위 K개의 박스만 반환
    """
    im_height, im_width, _ = img_raw.shape
    img = np.float32(img_raw)  # uint8에서 float32로 변환
    img -= (104, 117, 123)     # 이미지 정규화
    img = img.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    scale = torch.Tensor([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0]]).to(device)

    with torch.no_grad():
        loc, conf, landms = net(img)

    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)
    prior_data = priors.data

    # Decode boxes
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0, im_width)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, im_height)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, im_width)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, im_height)

    # Confidence scores
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    # Decode landmarks
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0],
                           img_raw.shape[1], img_raw.shape[0], img_raw.shape[1], img_raw.shape[0],
                           img_raw.shape[1], img_raw.shape[0]]).to(device)
    landms = landms * scale1
    landms = landms.cpu().numpy()

    # Top K filtering
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]

    # NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    dets = dets[keep, :]
    landms = landms[keep]

    return dets[:, :4], dets[:, 4], landms

def calculate_box_with_nose_and_ratios(landmarks, img_shape, previous_box=None, scale=1.0):
    """
    코를 중심으로 박스를 생성하며, 상단 좌표에서 눈까지와 하단 좌표에서 입까지의 비율을 유지합니다.
    또한, 이전 프레임의 크기를 반영합니다.
    """
    # 랜드마크를 (5, 2) 형태로 변환
    if len(landmarks.shape) == 1 and landmarks.size == 10:
        landmarks = landmarks.reshape(5, 2)

    # 코, 눈, 입 좌표
    nose = landmarks[2]
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    # 눈과 입의 평균 위치 계산
    eye_y = (left_eye[1] + right_eye[1]) / 2
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2

    # 이미지 높이에 대한 상단과 하단 비율 계산
    img_height, img_width = img_shape[:2]
    top_padding = int(img_height / 3)
    bottom_padding = int(img_height / 3)

    # 박스의 상단과 하단 좌표 결정
    box_top = max(0, int(nose[1] - top_padding))
    box_bottom = min(img_height, int(nose[1] + bottom_padding))

    # 박스의 높이와 너비 계산
    box_height = box_bottom - box_top
    box_width = int(box_height * scale)  # 정사각형 형태를 유지하기 위해 비율 적용

    # 박스의 좌우 좌표 계산
    box_left = max(0, int(nose[0] - box_width / 2))
    box_right = min(img_width, int(nose[0] + box_width / 2))

    # 이전 프레임 박스와 크기 동일하게 조정
    if previous_box is not None:
        prev_width = previous_box[2] - previous_box[0]
        prev_height = previous_box[3] - previous_box[1]
        center_x, center_y = nose
        box_left = max(0, int(center_x - prev_width / 2))
        box_right = min(img_width, int(center_x + prev_width / 2))
        box_top = max(0, int(center_y - prev_height / 2))
        box_bottom = min(img_height, int(center_y + prev_height / 2))

    return [box_left, box_top, box_right, box_bottom]

def detect_and_save_faces_with_landmark_tracking(video_path, output_path, embedding, face_detector, cfg, face_recognition, device='cuda'):
    cap = cv2.VideoCapture(video_path)
    previous_landmarks = None
    previous_box = None
    first_frame = True
    target_size = 640  # 정사각형 크기 (640x640)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (target_size, target_size))

    with tqdm(total=total_frames, desc="Processing Frames") as pbar:
        while True:
            ret, img = cap.read()
            if not ret:
                break

            if first_frame and (img.shape[1] > 1920 or img.shape[0] > 1080):
                img = resize_image(img)

            # RetinaFace 추론
            boxes, scores, landms = retinaface_inference(face_detector, img, cfg, device)

            top_k = 20
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landms = landms[order]
            scores = scores[order]

            if first_frame:
                if len(boxes) == 0:
                    print("No faces detected in the first frame. Skipping frame.")
                    pbar.update(1)
                    continue

                if embedding is not False:
                    annotations = [{'bbox': box, 'score': score, 'landmarks': landm} for box, score, landm in zip(boxes, scores, landms)]
                    best_face = select_best_face_based_on_embedding(face_recognition, annotations, embedding, img, img_shape=img.shape)
                else:
                    annotations = [{'bbox': box, 'score': score, 'landmarks': landm} for box, score, landm in zip(boxes, scores, landms)]
                    best_face = max(annotations, key=lambda face: (face['score'], (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1])))

                first_frame = False
            else:
                if len(boxes) == 0:
                    print("No faces detected. Skipping frame.")
                    pbar.update(1)
                    continue

                annotations = [{'bbox': box, 'score': score, 'landmarks': landm} for box, score, landm in zip(boxes, scores, landms)]
                best_face = get_most_similar_face(annotations, previous_landmarks)

            if best_face is None:
                print("No matching face found. Skipping frame.")
                pbar.update(1)
                continue

            # 랜드마크 및 박스 업데이트
            previous_landmarks = best_face['landmarks']
            box = calculate_square_box_with_landmarks(previous_landmarks, img.shape, previous_box)
            previous_box = box

            # 크롭된 얼굴 추출
            x1, y1, x2, y2 = map(int, box)
            cropped_face = img[y1:y2, x1:x2]

            # 원본 비율 유지하며 (640, 640)으로 리사이즈
            crop_height, crop_width = cropped_face.shape[:2]
            scale = target_size / max(crop_width, crop_height)
            resized_face = cv2.resize(cropped_face, (int(crop_width * scale), int(crop_height * scale)))

            # 중앙에 배치할 배경 생성
            avg_pixel_value = img.mean(axis=(0, 1), dtype=np.uint8)
            padded_frame = np.full((target_size, target_size, 3), avg_pixel_value, dtype=np.uint8)

            # 중앙에 삽입
            start_y = (target_size - resized_face.shape[0]) // 2
            start_x = (target_size - resized_face.shape[1]) // 2
            padded_frame[start_y:start_y + resized_face.shape[0], start_x:start_x + resized_face.shape[1]] = resized_face

            # 프레임 저장
            writer.write(padded_frame)
            pbar.update(1)

    cap.release()
    writer.release()
    print(f"Video saved at {output_path}")
    return True


def calculate_square_box_with_landmarks(landmarks, img_shape, previous_box=None, scale=2.8):
    """
    랜드마크를 기반으로 정사각형 박스를 계산합니다.
    가로/세로 중 큰 축에 맞춰 정사각형 박스를 생성하며, 이전 박스를 고려합니다.
    """
    # 랜드마크를 (5, 2) 형태로 변환
    if len(landmarks.shape) == 1 and landmarks.size == 10:
        landmarks = landmarks.reshape(5, 2)

    # 코, 눈, 입 좌표
    nose = landmarks[2]
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]

    # 중심점 계산 (코를 중심으로)
    center_x = int(nose[0])
    center_y = int((left_eye[1] + right_eye[1]) / 2)

    # 정사각형 크기 결정 (가로/세로 중 큰 축에 맞춤)
    eye_width = abs(right_eye[0] - left_eye[0])
    mouth_height = abs(right_mouth[1] - left_eye[1])
    box_size = int(max(eye_width, mouth_height) * scale)

    # 이전 박스와 크기 조정
    if previous_box is not None:
        prev_size = previous_box[2] - previous_box[0]
        box_size = int((box_size + prev_size) / 2)  # 이전 크기와 현재 크기의 평균

    # 정사각형 박스 좌표 계산
    x1 = max(0, center_x - box_size // 2)
    y1 = max(0, center_y - box_size // 2)
    x2 = min(img_shape[1], center_x + box_size // 2)
    y2 = min(img_shape[0], center_y + box_size // 2)

    return [x1, y1, x2, y2]
