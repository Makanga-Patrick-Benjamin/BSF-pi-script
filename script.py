import sqlite3
import time
from picamera2 import Picamera2
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
import locale

# Set locale to avoid warnings
locale.setlocale(locale.LC_ALL, 'C')

# --- Configuration ---
# Camera Settings
CAMERA_RESOLUTION = (1280, 720)  # Reduced resolution for better performance
CAMERA_WARMUP_TIME = 1.5  # seconds, allow camera to adjust

# Database Settings - Update this to your path
DATABASE_PATH = "/home/pato/Documents/sdf/text_ocr.db"
IMAGE_STORAGE_DIR = "/home/pato/Documents/sdf/ocr_images"

# EasyOCR Settings
EASYOCR_LANGUAGES = ['en']  # English language
MODEL_STORAGE = 'easyocr_models'  # Directory to store downloaded models

# Script Timing
CAPTURE_INTERVAL_SECONDS = 30  # Time between captures

# --- Initialize Hardware ---
picam2 = Picamera2()
try:
    camera_config = picam2.create_still_configuration(main={"size": CAMERA_RESOLUTION})
    picam2.configure(camera_config)
except Exception as e:
    print(f"Camera initialization error: {e}")
    exit(1)

# --- Initialize EasyOCR Reader ---
print("Initializing EasyOCR reader. This may download models on first run...")
try:
    reader = easyocr.Reader(
        EASYOCR_LANGUAGES,
        gpu=False,
        model_storage_directory=MODEL_STORAGE,
        download_enabled=True
    )
    print("EasyOCR reader initialized successfully.")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    print("Please ensure you have an internet connection for the first run.")
    exit(1)

# --- Database Functions ---
def initialize_database():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS extracted_text (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_content TEXT,
                image_path TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print(f"Database initialized at: {DATABASE_PATH}")
    except Exception as e:
        print(f"Database initialization error: {e}")
        exit(1)

def store_extracted_text(text, image_path, confidence):
    """Stores extracted text and metadata into the database."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO extracted_text (text_content, image_path, confidence)
            VALUES (?, ?, ?)
        ''', (text, image_path, confidence))
        conn.commit()
        conn.close()
        print(f"Stored text: '{text[:50]}...' (Confidence: {confidence:.2f}%)")
    except Exception as e:
        print(f"Database storage error: {e}")

# --- Image Processing Functions ---
def preprocess_image_for_digits(image):
    """Enhanced preprocessing specifically for digit recognition"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast_enhanced = clahe.apply(gray)
        
        # Thresholding
        _, thresh = cv2.threshold(contrast_enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Clean up noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    except Exception as e:
        print(f"Image preprocessing error: {e}")
        return None

# --- Text Extraction Function ---
def extract_digits_with_easyocr(image_path):
    """
    Extracts digits from an image using EasyOCR with:
    - Specialized preprocessing for digits
    - Post-processing to filter only numeric characters
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return None, 0

        processed_image = preprocess_image_for_digits(image)
        if processed_image is None:
            return None, 0

        extracted_digits = []
        all_confidences = []

        # Perform OCR
        results = reader.readtext(processed_image, detail=1)

        for (bbox, text, confidence) in results:
            # Extract only digits from the recognized text
            digits = ''.join([c for c in text if c.isdigit()])
            
            if digits:
                extracted_digits.append(digits)
                all_confidences.append(confidence)
                print(f"  Recognized digits '{digits}' with confidence: {confidence:.2f}")

        if extracted_digits:
            # Join all found digits with newlines
            final_text = "\n".join(extracted_digits)
            avg_confidence = (sum(all_confidences)/len(all_confidences)) * 100
            return final_text, avg_confidence
        else:
            return None, 0

    except Exception as e:
        print(f"OCR processing error: {e}")
        return None, 0

# --- Main Capture Function ---
def capture_and_process_image():
    """Main function to capture and process images"""
    try:
        # Create storage directory if needed
        os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        image_path = os.path.join(IMAGE_STORAGE_DIR, f"capture_{timestamp}.jpg")

        # Capture image
        print(f"\nCapturing image at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        picam2.start()
        time.sleep(CAMERA_WARMUP_TIME)
        picam2.capture_file(image_path)
        picam2.stop()
        print(f"Image saved to: {image_path}")

        # Extract digits
        text, confidence = extract_digits_with_easyocr(image_path)

        if text:
            print(f"\nExtracted digits (Confidence: {confidence:.2f}%):")
            print("=" * 40)
            print(text)
            print("=" * 40)
            store_extracted_text(text, image_path, confidence)
        else:
            print("\nNo digits detected in image.")
            os.remove(image_path)
            print(f"Removed image: {image_path}")

    except Exception as e:
        print(f"Capture/processing error: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting Digit OCR System")
    print("=" * 50)
    initialize_database()

    try:
        while True:
            capture_and_process_image()
            print(f"\nWaiting {CAPTURE_INTERVAL_SECONDS} seconds for next capture...")
            time.sleep(CAPTURE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nExiting program due to user interruption")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        picam2.close()
        print("Camera resources released.")
        print("Program terminated.")