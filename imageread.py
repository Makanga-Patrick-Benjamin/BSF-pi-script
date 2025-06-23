import sqlite3
import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np

# --- Configuration ---
# Database Settings (switch this to your path)
DATABASE_PATH = "/home/pato/Documents/sdf/text_ocr.db"
# Directory to store processed images (optional, you can just process in place)
# IMAGE_STORAGE_DIR = "/home/pato/Documents/sdf/ocr_processed_images"

# New: Directory containing images to be processed
INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/input_images" # <--- IMPORTANT: SET YOUR INPUT IMAGE FOLDER HERE!
PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/ocr_processed_images" # Directory to move processed images

# EasyOCR Settings
# For first run, EasyOCR will download models. This requires internet.
# Subsequent runs will use cached models.
EASYOCR_LANGUAGES = ['en'] # Languages to load. 'en' for English.

# --- Crucial for integer-only recognition ---
EASYOCR_ALLOWLIST = '0123456789'
EASYOCR_BLOCKLIST = '' # Keep blocklist empty when using allowlist

# Script Timing
PROCESS_INTERVAL_SECONDS = 10 # How often to check for new images and process them

# --- Initialize EasyOCR Reader ---
# This will download models on the first run if not present.
# Ensure you have an internet connection for the first execution.
print("Initializing EasyOCR reader. This may download models on first run...")
try:
    # Pass the allowlist to the reader initialization through recognizer_config
    reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=False, recognizer_config={'allowlist': EASYOCR_ALLOWLIST})
    print("EasyOCR reader initialized successfully for integer-only recognition.")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    print("Please ensure you have an internet connection for the first run to download models.")
    print("Also, check your EasyOCR installation. You might need to update it: pip install easyocr --upgrade")
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
    return denoised # Return denoised grayscale image

# --- Text Extraction Function (EasyOCR) ---
def extract_text_with_easyocr(image_path):
    """
    Extracts text from an image using EasyOCR, specifically filtering for integers.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, 0

    # Preprocess the image before passing to EasyOCR
    processed_image = preprocess_image_for_easyocr(image)
    if processed_image is None:
        return None, 0

    extracted_integers = []
    all_confidences = []

    try:
        # Perform OCR using EasyOCR. The allowlist is already configured in the reader.
        results = reader.readtext(processed_image)

        for (bbox, text, confidence) in results:
            # Step 1: Ensure the recognized text is not empty
            if text.strip():
                # Step 2: Filter out any non-digit characters that might still be present
                # (e.g., if OCR misinterprets a blurry dot as a part of the number)
                cleaned_text = ''.join(filter(str.isdigit, text.strip()))

                # Step 3: Check if the cleaned text is a valid integer
                if cleaned_text: # Ensure there are digits left after filtering
                    try:
                        # Convert to int to confirm it's a valid number and strip leading zeros
                        integer_value = int(cleaned_text)
                        extracted_integers.append(str(integer_value))
                        all_confidences.append(float(confidence))
                        print(f"  Recognized integer '{integer_value}' with confidence: {float(confidence):.2f}")
                    except ValueError:
                        # This block handles cases where cleaned_text might be empty or invalid after filtering
                        print(f"  Skipping non-integer segment after filtering: '{cleaned_text}' from original '{text}'")
                else:
                    print(f"  No digits found after filtering: '{text}'")
            else:
                print("  Skipping empty text detection.")

    except Exception as e:
        print(f"Error during EasyOCR processing: {e}")
        return None, 0

    if extracted_integers:
        final_extracted_text = "\n".join(extracted_integers)
        overall_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        return final_extracted_text, overall_avg_confidence * 100 # Convert to 0-100%
    else:
        return None, 0

# --- Main Processing Loop ---
def process_images_from_folder():
    """Main function to process images from a specified folder."""
    os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True) # Ensure processed directory exists

    print(f"\n--- Checking for new images in {INPUT_IMAGE_DIR} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    images_found = False
    for filename in os.listdir(INPUT_IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images_found = True
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            print(f"Processing image: {image_path}")

            # Extract text using EasyOCR, which is now configured for integers
            text, confidence = extract_text_with_easyocr(image_path)

            if text:
                print(f"\n--- Final Extracted Integers (Overall Confidence: {confidence:.2f}%) ---")
                print(text)
                store_extracted_text(text, image_path, confidence)
            else:
                print("\nNo integers detected in image by EasyOCR.")

            # Move the processed image to the PROCESSED_IMAGE_DIR
            destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
            os.rename(image_path, destination_path)
            print(f"Moved processed image to: {destination_path}")
    
    if not images_found:
        print("No new images found in the input folder.")

# --- Main Execution Block ---
if __name__ == "__main__":
    initialize_database()

    try:
        while True:
            process_images_from_folder()
            print(f"\nWaiting for {PROCESS_INTERVAL_SECONDS} seconds before checking again...")
            time.sleep(PROCESS_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nExiting program due to user interruption.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        print("Program finished.")