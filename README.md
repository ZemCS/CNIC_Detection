# CNIC Detection System

A comprehensive Pakistani CNIC (Computerized National Identity Card) detection and OCR system implemented with Python Flask backend and Flutter mobile application.

## Overview

This system provides automated detection and text extraction capabilities for Pakistani CNICs using computer vision and optical character recognition technologies. The solution consists of a Flask-based REST API backend and a cross-platform Flutter mobile application.

## Features

- YOLO-based CNIC detection with orientation classification
- Multi-language OCR text extraction using PaddleOCR and Tesseract
- Automatic image orientation correction
- Identity number validation between front and back sides
- Camera-based mobile interface with burst capture functionality
- Structured JSON data output
- Real-time image quality assessment using Laplacian variance

## System Requirements

### Backend Requirements
- Python 3.7 or higher
- OpenCV library
- Flask web framework
- Ultralytics YOLO
- PaddleOCR
- Pytesseract
- Tesseract OCR Engine

### Mobile Application Requirements
- Flutter SDK 3.0 or higher
- Dart 3.0 or higher
- Camera hardware access
- Network connectivity

## Installation

### Backend Configuration

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cnic-detection-system.git
   cd cnic-detection-system
   ```

2. Install required Python packages:
   ```bash
   pip install flask ultralytics opencv-python pytesseract paddleocr waitress numpy werkzeug
   ```

3. Install Tesseract OCR:
   - Windows: Download from the official Tesseract GitHub releases
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

4. Configure system paths in `cnic_api.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   MODEL_PATH = r"path\to\your\best.pt"
   OUTPUT_PATH = r"path\to\output.json"
   ```

5. Place the trained YOLO model file in the specified directory.

### Mobile Application Configuration

1. Navigate to the Flutter application directory:
   ```bash
   cd flutter_app
   ```

2. Install Flutter dependencies:
   ```bash
   flutter pub get
   ```

3. Configure the API endpoint in `OcrScreen.dart`:
   ```dart
   final String _apiUrl = 'http://SERVER_IP_ADDRESS:5000/cnic';
   ```

4. Add camera permissions:
   
   Android (`android/app/src/main/AndroidManifest.xml`):
   ```xml
   <uses-permission android:name="android.permission.CAMERA" />
   ```
   
   iOS (`ios/Runner/Info.plist`):
   ```xml
   <key>NSCameraUsageDescription</key>
   <string>Camera access required for CNIC scanning</string>
   ```

## Usage

### Server Deployment

Execute the following command to start the Flask server:
```bash
python cnic_api.py
```

The server will be accessible at `http://0.0.0.0:5000`.

### Mobile Application Execution

Launch the Flutter application:
```bash
flutter run
```

### API Reference

#### Health Check Endpoint
- **Method**: GET
- **URL**: `/home`
- **Response**: `{"message": "API is working"}`

#### CNIC Processing Endpoint
- **Method**: POST
- **URL**: `/cnic`
- **Parameters**:
  - `file_front`: Front side CNIC image (multipart/form-data)
  - `file_back`: Back side CNIC image (multipart/form-data)
- **Response**: JSON object containing extracted CNIC information

### Sample Response Format

```json
{
  "front": {
    "Name": "SAMPLE NAME",
    "Father Name": "FATHER NAME",
    "Gender": "M",
    "Identity Number": "12345-1234567-1",
    "Date of Birth": "01.01.1990",
    "Date of Issue": "01.01.2020",
    "Date of Expiry": "01.01.2030",
    "Country of Stay": "Pakistan",
    "isCNIC?": "True"
  },
  "back": {
    "Identity Number": "12345-1234567-1"
  }
}
```

## Technical Architecture

### YOLO Model Classes
The system utilizes the following classification schema:
- Standard orientation classes: `cnic_front`, `cnic_back`
- Rotational variants: `*_left`, `*_right`, `*_upside_down`
- Specialized class: `cnic_back_number` for identity number extraction

### Image Processing Pipeline
1. YOLO-based detection and classification
2. Bounding box extraction and cropping
3. Orientation correction based on classification
4. OCR text extraction using PaddleOCR
5. Structured data parsing and validation
6. JSON serialization and storage

### Mobile Application Architecture
1. Camera initialization and preview
2. Burst capture with quality assessment
3. Image preprocessing and enhancement
4. HTTP multipart request transmission
5. Response parsing and user interface rendering

## Configuration Parameters

### Detection Thresholds
- Confidence threshold: 0.5 minimum for CNIC detection
- Burst capture count: 2 images per side
- Image quality assessment: Laplacian variance calculation

### Supported Features
- Multi-orientation detection (0°, 90°, 180°, 270°)
- Bilateral language support (English and Urdu)
- International country recognition (195+ countries)
- Automatic image enhancement (contrast adjustment: 1.2x)

## Error Handling

The system implements comprehensive error handling for:
- Invalid file formats and corrupted images
- YOLO model loading failures
- OCR processing errors
- Identity number mismatch validation
- Network connectivity issues
- Camera access permissions

## Performance Considerations

- Optimized YOLO inference with GPU acceleration support
- Efficient memory management for image processing
- Asynchronous HTTP request handling
- Image compression for network transmission
- Caching mechanisms for model loading

## Project Structure

```
cnic-detection/
├── cnic_api.py              # Flask backend implementation
├── runs
│   ├── best.pt                  # YOLO model weights
├── output.json              # Processing output storage
├── cnic_detection_app/
│   ├── lib/
│   │   ├── main.dart        # Application entry point
│   │   └── OcrScreen.dart   # Camera interface implementation
│   └── pubspec.yaml         # Flutter dependencies
└── README.md               # Documentation
```

## Contributing

Contributions are welcome through the standard GitHub workflow:
1. Fork the repository
2. Create a feature branch
3. Implement changes with appropriate testing
4. Submit a pull request with detailed description

## License

This project is distributed under the MIT License. See the LICENSE file for complete terms and conditions.

## Support

For technical support and issue reporting, please utilize the GitHub issue tracking system. Ensure all system requirements are met and dependencies are properly configured before submitting support requests.
