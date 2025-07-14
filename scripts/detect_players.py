from ultralytics import YOLO
import cv2
import os

def detect_players(video_path, output_dir, model_path="best.pt", video_label=""):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    model = YOLO(model_path)
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            for i, box in enumerate(r.boxes):
                cls = int(box.cls[0])
                if cls == 0:  # 0 = player class
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    crop = frame[y1:y2, x1:x2]
                    save_path = os.path.join(output_dir, f"{video_label}_f{frame_id}_p{i}.jpg")
                    cv2.imwrite(save_path, crop)

        frame_id += 1

    cap.release()

if __name__ == "__main__":
    detect_players("broadcast.mp4", "crops/broadcast", model_path="best.pt", video_label="broadcast")
    detect_players("tacticam.mp4", "crops/tacticam", model_path="best.pt", video_label="tacticam")
