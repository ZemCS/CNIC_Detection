from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import pytesseract
import re
import json
import numpy as np
from paddleocr import PaddleOCR
from werkzeug.utils import secure_filename

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MODEL_PATH = r"C:\Users\lenovo\Programming\CNIC_Detection\runs\best.pt"
OUTPUT_PATH = r"C:\Users\lenovo\Programming\CNIC_Detection\output.json"
cnic_data = {'front': {}, 'back': {}}

import cv2
import numpy as np

def preprocess_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Estimate brightness
    mean_brightness = np.mean(gray)

    # Step 2: Adaptive CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    # Step 3: Adaptive denoising based on brightness
    if mean_brightness < 80:
        # Dim image – preserve detail, enhance contrast, minimal denoise
        h = 10
        alpha = 1.4
        beta = 30
    elif mean_brightness > 170:
        # Overexposed – reduce brightness, strong denoise
        h = 25
        alpha = 1.0
        beta = -20
    else:
        # Normal lighting
        h = 20
        alpha = 1.2
        beta = 15

    denoised = cv2.fastNlMeansDenoising(equalized, None, h=h, templateWindowSize=7, searchWindowSize=35)

    # Optional: median blur to remove fine noise (skip for dim images)
    if mean_brightness >= 80:
        denoised = cv2.medianBlur(denoised, 3)

    # Sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Brightness/contrast adjustment
    bright_contrast = cv2.convertScaleAbs(sharpened, alpha=alpha, beta=beta)

    # Optional: bilateral smoothing (skip for dim images)
    if mean_brightness >= 80:
        final = cv2.bilateralFilter(bright_contrast, d=7, sigmaColor=75, sigmaSpace=75)
    else:
        final = bright_contrast  # retain texture in dark images
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    alpha = 1.15
    beta = 11
    adjusted = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    cv2.imshow('Processed', adjusted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return adjusted


def extract_text(image: np.ndarray) -> str:
    # preprocessed_image = preprocess_image(image)
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    results = ocr.ocr(image)
    all_text = []
    for line in results:
        for res in line:
            txt, conf = res[1]
            all_text.append(txt)
            print(f"text: {txt}")
    return "\n".join(all_text)

def parse_cnic_text(text: str, class_name) -> dict:
    global cnic_data
    lines = text.split('\n')
    identity_re = re.compile(r'\d{5}-\d{7}-\d')
    date_re = re.compile(r'\d{2}\.\d{2}\.\d{4}')
    countries = [
        "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
        "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
        "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
        "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
        "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada",
        "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros",
        "Congo (Congo-Brazzaville)", "Costa Rica", "Croatia", "Cuba", "Cyprus", "Czechia",
        "Democratic Republic of the Congo", "Denmark", "Djibouti", "Dominica",
        "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea",
        "Eritrea", "Estonia", "Eswatini (fmr. Swaziland)", "Ethiopia", "Fiji", "Finland",
        "France", "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada",
        "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary",
        "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
        "Ivory Coast", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati",
        "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
        "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia",
        "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico",
        "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique",
        "Myanmar (formerly Burma)", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",
        "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman",
        "Pakistan", "Palau", "Palestine State", "Panama", "Papua New Guinea", "Paraguay",
        "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
        "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa",
        "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia",
        "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands",
        "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka",
        "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Tajikistan", "Tanzania",
        "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey",
        "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom",
        "United States of America", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City",
        "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
    ]
    if 'back' not in class_name:
        for line in lines:
            if identity_re.search(line):
                cnic_data['front']['Identity Number'] = identity_re.search(line).group()
            if date_re.search(line):
                date = date_re.search(line).group()
                if 'Date of Birth' not in cnic_data['front']:
                    cnic_data['front']['Date of Birth'] = date
                elif 'Date of Issue' not in cnic_data['front']:
                    cnic_data['front']['Date of Issue'] = date
                else:
                    cnic_data['front']['Date of Expiry'] = date
            if 'Name' in line and 'Name' not in cnic_data['front']:
                index = lines.index(line)
                cnic_data['front']['Name'] = lines[index + 1]
            if 'Father' in line and 'Father Name' not in cnic_data['front']:
                index = lines.index(line)
                cnic_data['front']['Father Name'] = lines[index + 1]
            if line.strip() in ['M', 'F']:
                cnic_data['front']['Gender'] = line
            for country in countries:
                if country in line and line != "PAKISTAN":
                    cnic_data['front']['Country of Stay'] = country
    else:
        for line in lines:
            if identity_re.search(line):
                cnic_data['back']['Identity Number'] = identity_re.search(line).group()
        
    return cnic_data

def save_to_json(cnic_data: dict, output_path: str) -> dict:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cnic_data, f, ensure_ascii=False, indent=4)
    return cnic_data

def extract_cnic_to_json(image: np.ndarray, output_path: str, class_name) -> dict:
    text = extract_text(image)
    cnic_data = parse_cnic_text(text, class_name)
    return save_to_json(cnic_data, output_path)

def crop_cnic(image: np.ndarray, box, class_name) -> dict:
    coords = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, coords)
    cropped_img = image[y1:y2, x1:x2]
    final_img = correct_orientation(cropped_img, class_name)
    return extract_cnic_to_json(final_img, OUTPUT_PATH, class_name)

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

@app.route('/home', methods=['GET'])
def home():
    return jsonify({'message': 'API is working'})

@app.route('/cnic', methods=['POST'])
def detect_cnic():
    if 'file_front' not in request.files:
        return jsonify({'error': 'CNIC front not provided'}), 400

    if 'file_back' not in request.files:
        return jsonify({"error": "CNIC back not provided"}), 400
    
    file_front = request.files['file_front']
    file_back = request.files['file_back']

    if file_front.filename == '' or file_back.filename == '':
        return jsonify({'error': 'No file selected'}), 200

    try:
        contents_front = file_front.read()
        nparr_front = np.frombuffer(contents_front, np.uint8)
        image_front = cv2.imdecode(nparr_front, cv2.IMREAD_COLOR)
        if image_front is None:
            return jsonify({'error': 'Invalid front image file.'}), 200
    except Exception as e:
        return jsonify({'error': f'Error reading front image: {str(e)}'}), 500

    try:
        contents_back = file_back.read()
        nparr_back = np.frombuffer(contents_back, np.uint8)
        image_back = cv2.imdecode(nparr_back, cv2.IMREAD_COLOR)
        if image_back is None:
            return jsonify({'error': 'Invalid back image file.'}), 200
    except Exception as e:
        return jsonify({'error': f'Error reading back image: {str(e)}'}), 500

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        return jsonify({'error': f'Error loading YOLO model: {str(e)}'}), 500

    try:
        results_front = model.predict(source=image_front, save=False)
        boxes_front = results_front[0].boxes

        if boxes_front is None or len(boxes_front) == 0:
            cnic_data = {'front': {'isCNIC?': 'False'}}
            save_to_json(cnic_data, OUTPUT_PATH)
            return jsonify(cnic_data)

        names = model.names
        for i, box in enumerate(boxes_front):
            cls_id = int(box.cls[0])
            class_name = names[cls_id]
            conf = float(box.conf[0])
            conf_threshold = 0.5

            if 'back' not in class_name.lower() and conf >= conf_threshold:    
                cnic_data = crop_cnic(image_front, box, class_name)
                results_back = model.predict(source=image_back, save=False)
                boxes_back = results_back[0].boxes

                if boxes_back is None or len(boxes_back) == 0:
                    cnic_data = {'back': {'isCNIC?': 'False'}}
                    save_to_json(cnic_data, OUTPUT_PATH)
                    return jsonify(cnic_data)
                
                matched = False
                for i, box in enumerate(boxes_back):
                    cls_id = int(box.cls[0])
                    class_name = names[cls_id]
                    conf = float(box.conf[0])
                    
                    if 'back' in class_name.lower() and conf >= conf_threshold:
                        if 'number' in class_name.lower():
                            cnic_data = crop_cnic(image_back, box, class_name)
                            if cnic_data['front']['Identity Number'] == cnic_data["back"]['Identity Number']:
                                cnic_data['front']['isCNIC?'] = 'True'
                                return jsonify(cnic_data)
                            else:
                                return jsonify({'error': 'Invalid CNIC, front and back identity numbers do not match.'}), 500
                        matched = True
                    else:
                        return jsonify({'error': f'Front of the CNIC sent into the front endpoint'}), 500
                if matched:
                    return jsonify({'error': 'CNIC back not readable'}), 500
                else:
                    return jsonify({'error': 'Back of the CNIC not detected properly'}), 500
            else:
                return jsonify({'error': f'Back of the CNIC sent into the front endpoint'}), 500
            
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)