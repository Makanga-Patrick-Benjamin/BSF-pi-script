import sqlite3
import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
from ultralytics import YOLO # Import YOLO
import requests

#--- configuration for the API  Endpoint---
API_ENDPOINT = "http://137.63.212.168:8001/api/data/"  # Replace with your actual API endpoint if needed


# --- Configuration ---
# Database Settings (switch this to your path)
# DATABASE_PATH = "/home/pato/Documents/sdf/BSF-pi-script/text_ocr.db"
# Directory to store processed images (optional, you can just process in place)
# IMAGE_STORAGE_DIR = "/home/pato/Documents/sdf/ocr_processed_images"

# New: Directory containing images to be processed
INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img" # <--- IMPORTANT: SET YOUR INPUT IMAGE FOLDER HERE!
PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/BSF-pi-script/ocr_processed_images" # Directory to move processed images

# EasyOCR Settings
# For first run, EasyOCR will download models. This requires internet.
# Subsequent runs will use cached models.
EASYOCR_LANGUAGES = ['en'] # Languages to load. 'en' for English.

# --- Crucial for integer-only recognition ---
EASYOCR_ALLOWLIST = '0123456789'
EASYOCR_BLOCKLIST = '' # Keep blocklist empty when using allowlist

# Script Timing
PROCESS_INTERVAL_SECONDS = 10 # How often to check for new images and process them

# YOLOv8 Model Configuration
YOLOV8_MODEL_PATH = "/home/pato/Documents/sdf/YoloRetrain.pt" # <--- IMPORTANT: SET PATH TO YOUR TRAINED YOLOv8 MODEL

# Calibration Factor (pixels per millimeter)
# YOU MUST CALIBRATE THIS FOR YOUR SPECIFIC CAMERA SETUP!
# To calibrate: Place an object of known real-world dimension (e.g., a ruler) in your image.
# Measure its length in pixels.
# PIXELS_PER_MM = (Measured Pixels) / (Known Millimeters)
# Example: If a 10mm object is 100 pixels, then PIXELS_PER_MM = 100/10 = 10
PIXELS_PER_MM = 20.0 # Placeholder: Adjust this value based on your calibration!

# --- Initialize EasyOCR Reader ---
# This will download models on the first run if not present.
# Ensure you have an internet connection for the first execution.
print("Initializing EasyOCR reader. This may download models on first run...")
try:
    # Pass the allowlist to the reader initialization through recognizer_config
    reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=False)
    print("EasyOCR reader initialized successfully for integer-only recognition.")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    print("Please ensure you have an internet connection for the first run to download models.")
    print("Also, check your EasyOCR installation. You might need to update it: pip install easyocr --upgrade")
    exit() # Exit if EasyOCR cannot be initialized

# --- Initialize YOLOv8 Model ---
print(f"Loading YOLOv8 model from: {YOLOV8_MODEL_PATH}...")
try:
    model = YOLO(YOLOV8_MODEL_PATH)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    print("Please ensure your model path is correct and Ultralytics is installed.")
    exit()

# --- Database Functions ---
# def initialize_database():
#     """Initializes the SQLite database and creates the table if it doesn't exist."""
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS extracted_text (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             text_content TEXT,
#             image_path TEXT,
#             confidence REAL,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
#         )
#     ''')

#     # Table for larvae measurements
#     cursor.execute('''
#         CREATE TABLE IF NOT EXISTS larvae_measurements (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             tray_number INTEGER,
#             larva_id INTEGER, -- Unique ID for each detected larva within a tray/image
#             length_mm REAL,
#             width_mm REAL,
#             area_sq_mm REAL,
#             estimated_weight_mg REAL,
#             confidence REAL,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#             FOREIGN KEY (tray_number) REFERENCES extracted_text(text_content) -- Link to tray number
#         )
#     ''')
#     conn.commit()
#     conn.close()
#     print(f"Database initialized at: {DATABASE_PATH}")

# def store_extracted_text(text, image_path, confidence):
#     """Stores extracted text and metadata into the database."""
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()
#     cursor.execute('''
#         INSERT INTO extracted_text (text_content, image_path, confidence)
#         VALUES (?, ?, ?)
#     ''', (text, image_path, confidence))
#     conn.commit()
#     conn.close()
#     print(f"Stored text: '{text[:50]}...' (Confidence: {confidence:.2f}%)")

# def store_larvae_data(tray_number, larva_id, length_mm, width_mm, area_sq_mm, estimated_weight_mg, confidence):
#     """Stores individual larva measurements into the database."""
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()
#     cursor.execute('''
#         INSERT INTO larvae_measurements (tray_number, larva_id, length_mm, width_mm, area_sq_mm, estimated_weight_mg, confidence)
#         VALUES (?, ?, ?, ?, ?, ?, ?)
#     ''', (tray_number, larva_id, length_mm, width_mm, area_sq_mm, estimated_weight_mg, confidence))
#     conn.commit()
#     conn.close()
#     print(f"Stored Larva {larva_id} (Tray {tray_number}): L={length_mm:.2f}mm, W={width_mm:.2f}mm, A={area_sq_mm:.2f}mm², Wt={estimated_weight_mg:.2f}mg")


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
        results = reader.readtext(processed_image, allowlist=EASYOCR_ALLOWLIST)

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
        # final_extracted_text = "\n".join(extracted_integers)
        final_extracted_text = extracted_integers[0]
        overall_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        return final_extracted_text, overall_avg_confidence * 100 # Convert to 0-100%
    else:
        return None, 0

# --- Larva Measurement and Weight Estimation ---
def calculate_larva_metrics(bbox, mask=None):
    """
    Calculates length, width, area in pixels from bounding box or mask,
    then converts to mm and estimates weight.
    """
    x1, y1, x2, y2 = bbox

    # Length and width from bounding box (in pixels)
    length_px = abs(y2 - y1) # Assuming larva is oriented vertically
    width_px = abs(x2 - x1)  # Assuming larva is oriented horizontally

    # Area calculation
    if mask is not None:
        # Use the mask to get a more accurate pixel area
        area_px = np.sum(mask) # Sum of pixels in the mask
    else:
        # Fallback to bounding box area if no mask (less accurate for irregular shapes)
        area_px = length_px * width_px

    # Convert pixel measurements to real-world measurements (mm)
    length_mm = length_px / PIXELS_PER_MM
    width_mm = width_px / PIXELS_PER_MM
    area_sq_mm = area_px / (PIXELS_PER_MM ** 2)

    # --- Weight Estimation ---
    # THIS IS A CRITICAL PART YOU NEED TO CALIBRATE!
    # A simple linear model (example: weight is proportional to area)
    # You will need to determine a more accurate model (e.g., polynomial regression)
    # based on actual measurements of your larvae.
    # Example: If 1 sq mm of larva area corresponds to X mg of weight.
    # You might find that larger larvae have a different density or shape.
    # For now, a very basic placeholder:
    # A common range for BSF larvae is 20-200mg.
    # Let's assume a simple average larva might be ~5mm long, ~1.5mm wide (roughly 7.5 sq mm area) and 50mg.
    # So, 50mg / 7.5 sq mm = ~6.67 mg/sq mm. This is a very rough estimate.
    # Consider using a more robust model like: weight = a * length_mm + b * width_mm + c
    # Or, weight = K * (area_sq_mm ** exponent)
    WEIGHT_PER_SQ_MM = 6.67 # Placeholder: Adjust this based on your empirical data!
    estimated_weight_mg = area_sq_mm * WEIGHT_PER_SQ_MM

    return length_mm, width_mm, area_sq_mm, estimated_weight_mg


# --- Main Processing Loop ---
def process_images_from_folder():
    """Main function to process images from a specified folder and send data to FastAPI."""
    os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True) # Ensure processed directory exists

    print(f"\n--- Checking for new images in {INPUT_IMAGE_DIR} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    images_found = False
    for filename in os.listdir(INPUT_IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images_found = True
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            print(f"Processing image: {image_path}")

            # 1. Extract Tray Number using EasyOCR
            tray_number_str, ocr_confidence = extract_text_with_easyocr(image_path)
            if tray_number_str:
                try:
                    tray_number = int(tray_number_str)
                    print(f"Detected Tray Number: {tray_number} (Confidence: {ocr_confidence:.2f}%)")
                    # No longer storing OCR text directly here, it's just for the tray_number
                except ValueError:
                    print(f"Warning: Could not convert '{tray_number_str}' to an integer for tray number. Skipping larvae analysis for this image.")
                    tray_number = None
            else:
                print("No tray number detected by EasyOCR. Skipping larvae analysis for this image.")
                tray_number = None

            if tray_number is None:
                destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
                os.rename(image_path, destination_path)
                print(f"Moved image (no tray number): {image_path} to {destination_path}")
                continue

            # 2. Perform Larvae Detection and Measurement using YOLOv8
            print(f"Running YOLOv8 inference on {image_path}...")
            larvae_data_to_send = [] # Collect all larva data for this image
            total_count = 0
            
            try:
                yolo_results = model(image_path)

                for result in yolo_results:
                    if result.boxes:
                        total_count = len(result.boxes)
                        print(f"Found {total_count} larvae in Tray {tray_number}.")

                        for larva_id, box in enumerate(result.boxes):
                            bbox_xyxy = box.xyxy[0].tolist()
                            larva_confidence = box.conf[0].item()

                            mask = None
                            if result.masks and len(result.masks.data) > larva_id:
                                mask = result.masks.data[larva_id].cpu().numpy()
                                original_h, original_w, _ = cv2.imread(image_path).shape
                                mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)

                            length_mm, width_mm, area_sq_mm, estimated_weight_mg = \
                                calculate_larva_metrics(bbox_xyxy, mask)

                            # Prepare data for sending
                            larvae_data_to_send.append({
                                "tray_number": tray_number,
                                "length": round(length_mm, 2), # Round for sending
                                "width": round(width_mm, 2),
                                "area": round(area_sq_mm, 2),
                                "weight": round(estimated_weight_mg, 2),
                                "count": 1 # Each entry is for one larva, total count is summed up later
                            })
                            print(f"  Larva {larva_id + 1}: L={length_mm:.2f}mm, W={width_mm:.2f}mm, A={area_sq_mm:.2f}mm², Wt={estimated_weight_mg:.2f}mg (Conf: {larva_confidence:.2f}%)")
                    else:
                        print(f"No larvae detected by YOLOv8 in Tray {tray_number}.")

                # 3. Send data to FastAPI endpoint
                if larvae_data_to_send:
                    # For simplicity, we'll send each larva's data individually.
                    # Alternatively, you could aggregate and send combined metrics and total count.
                    # If you send individual, `count` will always be 1 for each larva.
                    # The `LarvaeData` model in Flask has a `count` column; if you store individual
                    # larvae, that `count` would be 1. If you store aggregated per tray, it's the total.
                    # The current Flask model seems designed for aggregated data (one entry per timestamp per tray).
                    # Let's aggregate the data per tray before sending to match the Flask model.

                    if total_count > 0:
                        avg_length = sum(d['length'] for d in larvae_data_to_send) / total_count
                        avg_width = sum(d['width'] for d in larvae_data_to_send) / total_count
                        avg_area = sum(d['area'] for d in larvae_data_to_send) / total_count
                        avg_weight = sum(d['weight'] for d in larvae_data_to_send) / total_count
                        
                        payload = {
                            "tray_number": tray_number,
                            "length": round(avg_length, 2),
                            "width": round(avg_width, 2),
                            "area": round(avg_area, 2),
                            "weight": round(avg_weight, 2),
                            "count": total_count # Total count for this tray
                        }

                        print(f"Sending aggregated data for Tray {tray_number} to FastAPI: {payload}")
                        try:
                            response = requests.post(API_ENDPOINT, json=payload)
                            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                            print(f"Data sent successfully to FastAPI. Response: {response.json()}")
                        except requests.exceptions.RequestException as req_e:
                            print(f"Error sending data to FastAPI: {req_e}")
                            print(f"Response content: {req_e.response.text if req_e.response else 'N/A'}")
                    else:
                        print(f"No larvae detected for Tray {tray_number}. No data sent to FastAPI.")

            except Exception as e:
                print(f"Error during YOLOv8 inference or data aggregation for {image_path}: {e}")

            # Move the processed image
            destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
            os.rename(image_path, destination_path)
            print(f"Moved processed image: {image_path} to {destination_path}")
    
    if not images_found:
        print("No new images found in the input folder.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # initialize_database()

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