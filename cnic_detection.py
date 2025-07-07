from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import cv2
import pytesseract
from PIL import Image
import re
import json
import numpy as np
from paddleocr import PaddleOCR

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

was_front_found = False
output_path = r'C:\Users\lenovo\Programming\CNIC_Detection\output.json'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    alpha = 1.5
    beta = 15
    adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(adjusted, -1, kernel)
    sharpened = cv2.fastNlMeansDenoising(sharpened, None, h=18, templateWindowSize=8, searchWindowSize=24)
    plt.figure(figsize=(10, 6))
    plt.imshow(sharpened, cmap='gray')
    plt.title('Preprocessed Image')
    plt.axis('off')
    plt.show()
    return sharpened

def extract_text(image):
    preprocessed_image = preprocess_image(image)
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    results = ocr.ocr(preprocessed_image)

    all_text = []
    for line in results:
        for res in line:
            txt, conf = res[1]
            print(f'Text: {txt}, {conf}')
            all_text.append(txt)

    full_text = "\n".join(all_text)
    return full_text

def parse_cnic_text(text):
    global was_front_found
    
    lines = text.split('\n')
    cnic_data = {'english': {}}

    identity_re = re.compile(r'\d{5}-\d{7}-\d')
    date_re = re.compile(r'\d{2}\.\d{2}\.\d{4}')

    for line in lines:
        if identity_re.search(line):
            cnic_data['english']['Identity Number'] = identity_re.search(line).group()
        
        if date_re.search(line):
            date = date_re.search(line).group()
            if 'Date of Birth' not in cnic_data['english']:
                cnic_data['english']['Date of Birth'] = date
            elif 'Date of Issue' not in cnic_data['english']:
                cnic_data['english']['Date of Issue'] = date
            else:
                cnic_data['english']['Date of Expiry'] = date
        
        if 'Name' in line and 'Name' not in cnic_data['english']:
            index = lines.index(line)
            cnic_data['english']['Name'] = lines[index + 1]
        
        if 'Father' in line and 'Father Name' not in cnic_data['english']:
            index = lines.index(line)
            cnic_data['english']['Father Name'] = lines[index + 1]

        if line.strip() == 'M' or line.strip() == 'F':
            cnic_data['english']['Gender'] = line

        if 'Country of Stay' in line:
            index = lines.index(line)
            cnic_data['english']['Country of Stay'] = lines[index + 2]

        cnic_data['english']['isCNIC?'] = 'True'
    return cnic_data
            
    
def save_to_json(cnic_data, output_path):
    global was_front_found
    if not was_front_found:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cnic_data, f, ensure_ascii=False, indent=4)
            was_front_found = True

def extract_cnic_to_json(image, output_path):
    text = extract_text(image)
    cnic_data =  parse_cnic_text(text)
    save_to_json(cnic_data, output_path)

def crop_cnic(image, box):
    coords = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, coords)
    cropped_img = image[y1:y2, x1:x2]
    final_img = correct_orientation(cropped_img)
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.title('Cropped CNIC')
    plt.axis('off')
    plt.show()
    extract_cnic_to_json(final_img, output_path)
    return final_img

def correct_orientation(image: np.ndarray, class_name) -> np.ndarray:
    if 'back' not in class_name:
        if 'left' in class_name:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif 'right' in class_name:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif 'upside_down' in class_name:
            image = cv2.rotate(image, cv2.ROTATE_180)
    else:
        if 'left' in class_name:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif 'right' in class_name:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif 'upside_down' in class_name:
            image = cv2.rotate(image, cv2.ROTATE_180)
    return image

def predict_cnic(image_path, model_path, conf_threshold=0.5):
    
    global was_front_found

    if not os.path.exists(image_path):
        print("Image not found")
        return
    
    model = YOLO(model_path)

    results = model.predict(source=image_path, save=False)

    names = model.names
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        print("No CNIC detected")
        save_to_json({'english': {'isCNIC?': 'False'}}, output_path)
        return
    
    if not was_front_found:

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            class_name =  names[cls_id]
            conf = float(box.conf[0])

            if conf >= conf_threshold:
                print(f"CNIC detected with {conf:.2f} confidence")

                plotted_image = results[0].plot()
                
                plt.figure(figsize=(10,6))
                plt.imshow(cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB))
                plt.title('Detected CNIC')
                plt.axis('off')
                plt.show()

                image = cv2.imread(image_path)
                crop_cnic(image, boxes[0])
    
    if was_front_found:
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            class_name =  names[cls_id]
            conf = float(box.conf[0])

            if class_name.lower() == 'cnic back' and conf >= conf_threshold:
                print(f'CNIC Back detected with {conf:.2f} confidence')

                plotted_image = results[0].plot()

                plt.figure(figsize=(10,6))
                plt.imshow(cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB))
                plt.title('Detected CNIC Back')
                plt.axis('off')
                plt.show()

                image = cv2.imread(image_path)
                crop_cnic(image, boxes[0])


if __name__ == "__main__":
    image_path = r"c:\Users\lenovo\Programming\CNIC_Detection\tests\cnic_back_ud.jpg"
    model_path = r"C:\Users\lenovo\Programming\CNIC_Detection\runs\best.pt"
    predict_cnic(image_path, model_path)