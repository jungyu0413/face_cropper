import face_recognition
import pandas as pd
import os
from retinaface.pre_trained_models import get_model
from utils_face_crop import *


def main():
    video_df = pd.read_csv('/workspace/ABAW/csv_annotation/train.csv')  # Load video paths
    sample_non_dt = []
    non_face_dt = []
    face_detector = get_model("resnet50_2020-07-20", max_size=2048)
    face_detector.eval()

    for _, row in video_df[14:15].iterrows():
        sample_img_path = row['path'].split('.')[0].replace('orgin_data', 'cropped_data') + '/00001.jpg'
        vid_path = row['path']
        save_path = row['path'].replace('orgin_data', 'face_data')

        # 파일 존재 여부 확인
        if not os.path.exists(sample_img_path):
            print(f"Sample image not found: {sample_img_path}. Skipping this sample.")
            sample_non_dt.append(row)
            continue

        try:
            # 샘플 얼굴 로드 및 임베딩 생성
            img = face_recognition.load_image_file(sample_img_path)
            embedding = face_recognition.face_encodings(img)[0]
        except Exception as e:
            print(f"Error processing sample image: {sample_img_path}, {e}. Skipping embedding.")
            sample_non_dt.append(row)
            embedding = False

        try:
            # 비디오 처리
            face_detected = detect_and_save_faces_with_landmark_tracking(vid_path, save_path, embedding, face_detector, face_recognition)
        except Exception as e:
            print(f"Error processing video: {vid_path}, {e}. Skipping video.")
            non_face_dt.append(row)
            continue

        if not face_detected:
            non_face_dt.append(row)

    # Save non-detected samples and faces to CSV
    non_sample_df = pd.DataFrame(sample_non_dt)
    non_face_df = pd.DataFrame(non_face_dt)
    os.makedirs('/workspace/ABAW/non_detect_csv', exist_ok=True)
    non_sample_df.to_csv('/workspace/ABAW/non_detect_csv/non_sample.csv', index=False)
    non_face_df.to_csv('/workspace/ABAW/non_detect_csv/non_face.csv', index=False)

if __name__ == '__main__':
    main()
