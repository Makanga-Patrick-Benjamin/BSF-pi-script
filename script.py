import sqlite3
import time
from picamera2 import Picamera2
import cv2
import os
import easyocr 
from datetime import datetime
import numpy as np

# --- Configuration ---
# Camera Settings
CAMERA_RESOLUTION = (1920, 1080) # Increased resolution for better text detail
CAMERA_WARMUP_TIME = 2 # seconds, allow camera to adjust
# Optional: Manual camera controls for consistent lighting (uncomment and tune if needed)
# CAMERA_EXPOSURE_TIME = 10000 # Microseconds
# CAMERA_ANALOGUE_GAIN = 1.0    # 1.0 is lowest gain

# Database Settings swicth this to your path
DATABASE_PATH = "/home/pato/Documents/sdf/text_ocr.db"
IMAGE_STORAGE_DIR = "/home/pato/Documents/sdf/ocr_images"

# EasyOCR Settings
# Languages to load. 'en' for English.
# For first run, EasyOCR will download models. This requires internet.
# Subsequent runs will use cached models.
EASYOCR_LANGUAGES = ['en']
# For faster inference on Raspberry Pi, you might limit recognition to
# specific character sets if you know your text is constrained (e.g., numbers).
# This is an advanced optimization and often not needed for general text.
# EASYOCR_ALLOWLIST = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# EASYOCR_BLOCKLIST = ''

# Script Timing
CAPTURE_INTERVAL_SECONDS = 30 # How often to capture and process an image this can be adjusted based on your needs

# --- Initialize Hardware ---
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": CAMERA_RESOLUTION})
picam2.configure(camera_config)

# --- Initialize EasyOCR Reader ---
# This will download models on the first run if not present.
# Ensure you have an internet connection for the first execution.
print("Initializing EasyOCR reader. This may download models on first run...")
try:
    reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=False) # gpu=False for Raspberry Pi CPU     pi 5 is more ideal for this
    # If you want to use GPU, set gpu=True but ensure you have the necessary setup.
    # reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=True) # Uncomment if using GPU
    print("EasyOCR reader initialized successfully.")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    print("Please ensure you have an internet connection for the first run to download models.")
    print("Also, check easyocr installation: pip install easyocr")
    exit() # Exit if EasyOCR cannot be initialized

# --- Database Functions ---
def initialize_database():
    """Initializes the SQLite database and creates the table if it doesn't exist."""
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

def store_extracted_text(text, image_path, confidence):
    """Stores extracted text and metadata into the database."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO extracted_text (text_content, image_path, confidence)
        VALUES (?, ?, ?)
    ''', (text, image_path, confidence))
    conn.commit()
    conn.close()
    print(f"Stored text: '{text[:50]}...' (Confidence: {confidence:.2f}%)")

# --- Image Preprocessing for EasyOCR ---
def preprocess_image_for_easyocr(image):
    """
    Applies basic preprocessing suitable for EasyOCR.
    EasyOCR is quite robust, so heavy preprocessing is often not needed.
    """
    if image is None or image.size == 0:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optional: Denoising - median blur can help with general noise
    denoised = cv2.medianBlur(gray, 3) # Kernel size 3

    # EasyOCR often handles binarization internally, but you can experiment
    # with adaptive thresholding if results are poor on original images.
    # _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # return thresh
    return denoised # Return denoised grayscale image

# --- Text Extraction Function (EasyOCR) ---
def extract_text_with_easyocr(image_path):
    """
    Extracts text from an image using EasyOCR.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, 0

    # Preprocess the image before passing to EasyOCR
    processed_image = preprocess_image_for_easyocr(image)
    if processed_image is None:
        return None, 0

    extracted_texts_list = []
    all_confidences = []

    try:
        # Perform OCR using EasyOCR
        # reader.readtext returns a list of detections:
        # [(bbox, text, confidence), (bbox, text, confidence), ...]
        # You can add parameters like detail=0 for less detailed output (just text)
        # or paragraph=True to merge lines into paragraphs.
        # Here, we process individual detections.
        results = reader.readtext(processed_image)

        for (bbox, text, confidence) in results:
            if text.strip(): # Ensure text is not empty
                extracted_texts_list.append(text.strip())
                all_confidences.append(float(confidence))
                print(f"  Recognized '{text.strip()}' with confidence: {float(confidence):.2f}")

    except Exception as e:
        print(f"Error during EasyOCR processing: {e}")
        return None, 0

    if extracted_texts_list:
        final_extracted_text = "\n".join(extracted_texts_list)
        # Average confidence from all detected text blocks
        overall_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        return final_extracted_text, overall_avg_confidence * 100 # Convert to 0-100%
    else:
        return None, 0

# --- Main Capture and Process Loop ---
def capture_and_process_text():
    """Main function to capture an image and process text using EasyOCR."""
    os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True) # Ensure image storage directory exists

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    image_path = os.path.join(IMAGE_STORAGE_DIR, f"capture_{timestamp}.jpg")

    print(f"\n--- Capturing image at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
    picam2.start()
    time.sleep(CAMERA_WARMUP_TIME) # Allow camera to adjust
    # Set manual controls if uncommented in config
    # picam2.set_controls({"ExposureTime": CAMERA_EXPOSURE_TIME, "AnalogueGain": CAMERA_ANALOGUE_GAIN})
    picam2.capture_file(image_path)
    picam2.stop()
    print(f"Image captured to: {image_path}")

    # Extract text using EasyOCR
    text, confidence = extract_text_with_easyocr(image_path)

    if text:
        print(f"\n--- Final Extracted Text (Overall Confidence: {confidence:.2f}%) ---")
        print(text)
        store_extracted_text(text, image_path, confidence)
    else:
        print("\nNo text detected in image by EasyOCR.")
        # Optionally delete the image if no text was found, to save space
        os.remove(image_path)
        print(f"Removed image: {image_path}")

# --- Main Execution Block ---
if __name__ == "__main__":
    initialize_database()

    try:
        while True:
            capture_and_process_text()
            print(f"\nWaiting for {CAPTURE_INTERVAL_SECONDS} seconds...")
            time.sleep(CAPTURE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nExiting program due to user interruption.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        picam2.close()
        print("PiCamera2 closed.")
        print("Program finished.")