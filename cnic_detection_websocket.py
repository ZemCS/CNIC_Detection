import asyncio
import websockets
import cv2
import base64
import numpy as np
import json
from ultralytics import YOLO

model = YOLO(r'C:\Users\lenovo\Programming\CNIC_Detection\runs\detect\best.pt')

async def detect_and_respond(websocket):
    async for message in websocket:
        frame_data = base64.b64decode(message)
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model.predict(source=frame, save=False)
        boxes = results[0].boxes
        names = model.names

        detections = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = names[cls_id]
            if conf >= 0.5 and class_name.lower().startswith('cnic'):
                coords = box.xyxy[0].cpu().numpy().tolist()
                detections.append({
                    'class': class_name,
                    'confidence': round(conf, 2),
                    'box': [int(c) for c in coords]
                })

        await websocket.send(json.dumps(detections))

async def main():
    async with websockets.serve(detect_and_respond, 'localhost', 8765):
        print('Websocket server started on ws://localhost:8765')
        await asyncio.Future()

if __name__ == '__main__':
    asyncio.run(main())