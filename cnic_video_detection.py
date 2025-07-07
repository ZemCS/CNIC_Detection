import cv2
from ultralytics import YOLO

model_path = r"C:\Users\lenovo\Programming\CNIC_Detection\runs\detect\best.pt"
model = YOLO(model_path)

video_path = r"C:\Users\lenovo\Programming\CNIC_Detection\tests\video_test.mp4"
output_path = r"C:\Users\lenovo\Programming\CNIC_Detection\tests\output_annotated.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
conf_threshold = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=conf_threshold, save=False)
    boxes = results[0].boxes
    names = model.names

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = names[cls_id]

            if class_name.lower() in ["cnic", "cnic back"] and conf >= conf_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)

    out.write(frame)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"Processed frame: {frame_idx}")

cap.release()
out.release()
print(f"Video saved at {output_path}")