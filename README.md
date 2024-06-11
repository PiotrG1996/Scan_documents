# Document Scanner

This project is a document scanner that uses edge detection and perspective transformation to scan documents from images or a webcam feed. It processes the image to detect the document edges, warps the perspective to get a top-down view, and applies adaptive thresholding to create a clean, scanned document look.

## Features

- Edge detection using Canny
- Contour detection to find document edges
- Perspective transformation to get a top-down view of the document
- Adaptive thresholding for a clean scanned document appearance
- Save scanned images with a single key press

## Installation

### Prerequisites

- Python 3.x
- OpenCV
- NumPy

### Steps

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/document-scanner.git
    cd document-scanner
    ```

2. Create a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### `requirements.txt`




```text
opencv-python
numpy

document-scanner/
├── main.py
├── utlis.py
├── requirements.txt
└── README.md
```

## Usage

```bash
# USB Camera
python main.py --webcam

# File
python main.py --file path/to/your/image.jpg
```

