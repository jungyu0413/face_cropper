import os
import cv2
import pandas as pd
from tqdm import tqdm


output_video_dir = "/workspace/ABAW/filtered_videos"
output_annotation_dir = "/workspace/ABAW/filtered_annotations"
valid_labels = set([0, 1, 2, 3, 4, 5, 6])
video_df = pd.read_csv('/workspace/ABAW/csv_annotation/face_data/train.csv') 
annotation_dir = "/workspace/ABAW/Third ABAW Annotations/EXPR_Classification_Challenge/Train_Set_refinement"
video_df['annotation_path'] = video_df['vid_name'].apply(lambda x: os.path.join(annotation_dir, f"{x}.txt"))

# 출력 디렉토리 생성
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_annotation_dir, exist_ok=True)

for _, row in tqdm(video_df.iterrows(), total=len(video_df), desc="Processing videos"):
    video_path = row['path']
    annotation_path = row['annotation_path']
    vid_name = row['vid_name']

    # Annotation 파일 확인
    if not os.path.exists(annotation_path):
        print(f"Annotation file not found for {vid_name}. Skipping...")
        continue

    # Annotation 로드
    annotations = pd.read_csv(annotation_path)

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}. Skipping...")
        continue

    # 비디오 메타데이터 가져오기
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 필터링된 비디오 저장 경로
    output_video_path = os.path.join(output_video_dir, f"{vid_name}.mp4")
    filtered_annotation_path = os.path.join(output_annotation_dir, f"{vid_name}.csv")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # 필터링된 어노테이션을 저장할 리스트 초기화
    filtered_annotations = []

    # Annotation에 따라 프레임 필터링
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 프레임의 라벨 가져오기
        if frame_idx >= len(annotations):
            break
        label = annotations.iloc[frame_idx]['label']

        # 유효한 레이블인지 확인
        if label in valid_labels:
            out.write(frame)  # 유효한 레이블의 프레임만 저장
            filtered_annotations.append(label)

        frame_idx += 1

    # 리소스 해제
    cap.release()
    out.release()
    # 필터링된 어노테이션 저장
    if filtered_annotations:
        pd.DataFrame(filtered_annotations).to_csv(filtered_annotation_path, index=False)
        print(f"Filtered annotation saved at: {filtered_annotation_path}")
    else:
        print(f"No valid annotations found for {vid_name}. Skipping annotation save.")

    print(f"Filtered video saved at: {output_video_path}")

print("Processing complete!")