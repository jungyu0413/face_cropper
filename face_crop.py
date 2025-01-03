import face_recognition
import pandas as pd
import os
from retinaface.pre_trained_models import get_model
from utils_face_crop import *
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace


def main():
    video_df = pd.read_csv('/workspace/ABAW/csv_annotation/train.csv')
    sample_non_dt = []
    non_face_dt = []

    # 모델 로드
    net = RetinaFace(cfg=cfg_re50, phase ='test')
    face_detector, cfg = load_retinaface_model(net, '/workspace/MAE/Pytorch_Retinaface/weights/Resnet50_Final.pth')
    face_detector.eval()

    for _, row in video_df.iterrows():
        sample_img_path = row['path'].split('.')[0].replace('orgin_data', 'cropped_data') + '/00001.jpg'
        vid_path = row['path']
        save_path = row['path'].replace('orgin_data', 'face_data')

        try:
            img = face_recognition.load_image_file(sample_img_path)
            embedding = face_recognition.face_encodings(img)[0]
        except Exception as e:
            print(f"Error processing sample image: {sample_img_path}, {e}. Skipping embedding.")
            sample_non_dt.append(row)
            embedding = False

        try:
            face_detected = detect_and_save_faces_with_landmark_tracking(vid_path, save_path, embedding, face_detector, cfg, face_recognition)
        except Exception as e:
            print(f"Error processing video: {vid_path}, {e}. Skipping video.")
            non_face_dt.append(row)
            continue

        if not face_detected:
            non_face_dt.append(row)

    pd.DataFrame(sample_non_dt).to_csv('/workspace/ABAW/non_detect_csv/non_sample.csv', index=False)
    pd.DataFrame(non_face_dt).to_csv('/workspace/ABAW/non_detect_csv/non_face.csv', index=False)


if __name__ == '__main__':
    main()