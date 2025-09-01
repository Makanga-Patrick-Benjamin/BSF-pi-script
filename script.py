import time
import cv2
import os
import easyocr
from datetime import datetime, timezone
import numpy as np
import paho.mqtt.client as mqtt # Import MQTT library
import json # To send data as JSON
import requests
import base64
import traceback # Added for better error logging
import sys

# --- Flat-Bug Model Imports ---
from flat_bug.predictor import Predictor
from flat_bug.config import DEFAULT_CFG, read_cfg
from flat_bug import logger as flatbug_logger, set_log_level

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "bsf_monitor/larvae_data"
                                      
# --- Flask Server Configuration ---
FLASK_SERVER_URL = "https://soldierfly-fly-monitor.onrender.com"  # Change this to your Flask server's URL if it's not local

# --- Callbacks for MQTT Client ---
def on_connect(client, userdata, flags, rc, properties):
    """Callback function for when the MQTT client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

# --- Configuration ---
INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img" # IMPORTANT: SET YOUR INPUT IMAGE FOLDER HERE!
PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/processed_images"
OUTPUT_DETECTION_DIR = "/home/pato/Documents/sdf/BSF-pi-script/detected_images"

# EasyOCR Settings
EASYOCR_LANGUAGES = ['en']
EASYOCR_ALLOWLIST = '0123456789'
EASYOCR_BLOCKLIST = ''

# Script Timing
PROCESS_INTERVAL_SECONDS = 10

# Flat-Bug Model Configuration
FLATBUG_MODEL_PATH = "/home/pato/Documents/sdf/best.pt" # IMPORTANT: SET PATH TO YOUR DOWNLOADED FLAT-BUG MODEL WEIGHTS (.pt file)
FLATBUG_DEVICE = "cpu"
FLATBUG_DTYPE = "float32"

# Calibration Factor (pixels per millimeter)
PIXELS_PER_MM = 20.0

# --- Initialize EasyOCR Reader ---
print("Initializing EasyOCR reader. This may download models on first run...")
try:
    reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=False)
    print("EasyOCR reader initialized successfully for integer-only recognition.")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    print("Please ensure you have an internet connection for the first run to download models.")
    exit()

# --- Initialize Flat-Bug Model ---
print(f"Loading Flat-Bug model from: {FLATBUG_MODEL_PATH} on device: {FLATBUG_DEVICE}...")
try:
    flatbug_config = DEFAULT_CFG
    flatbug_predictor = Predictor(
        FLATBUG_MODEL_PATH,
        device=FLATBUG_DEVICE,
        dtype=FLATBUG_DTYPE,
        cfg=flatbug_config
    )
    print("Flat-Bug model loaded successfully.")
except Exception as e:
    print(f"Error loading Flat-Bug model: {e}")
    print("Please ensure your model path is correct and flat-bug library is installed.")
    exit()

# --- Initialize MQTT Client ---
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start() 
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")
    exit()

# --- Image Preprocessing for EasyOCR ---
def preprocess_image_for_easyocr(image):
    """Converts image to grayscale and applies median blur for OCR."""
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3) 
    return denoised

# --- Text Extraction Function (EasyOCR) ---
def extract_text_with_easyocr(image_path):
    """
    Extracts integer text (assumed to be tray number) from an image using EasyOCR.
    Returns the extracted integer as a string and its confidence.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None, 0

    processed_image = preprocess_image_for_easyocr(image)
    if processed_image is None:
        return None, 0

    extracted_integers = []
    all_confidences = []

    try:
        # Use a simplified approach without the `reader` variable which is initialized in the main body
        results = reader.readtext(processed_image, allowlist=EASYOCR_ALLOWLIST)

        for (bbox, text, confidence) in results:
            if text.strip():
                cleaned_text = ''.join(filter(str.isdigit, text.strip()))
                if cleaned_text:
                    try:
                        integer_value = int(cleaned_text)
                        extracted_integers.append(str(integer_value))
                        all_confidences.append(float(confidence))
                    except ValueError:
                        print(f"  Skipping non-integer segment after filtering: '{cleaned_text}' from original '{text}'")
                else:
                    print(f"  No digits found after filtering: '{text}'")
            else:
                print("  Skipping empty text detection.")

    except Exception as e:
        print(f"Error during EasyOCR processing: {e}")
        return None, 0

    if extracted_integers:
        final_extracted_text = extracted_integers[0]
        overall_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        return final_extracted_text, overall_avg_confidence * 100
    else:
        return None, 0

# --- Larva Measurement and Weight Estimation ---
def calculate_larva_metrics(bbox, mask=None):
    """
    Calculates larva length, width, area, and estimated weight based on bounding box and mask.
    Assumes bbox is [x1, y1, x2, y2] in pixels.
    """
    x1, y1, x2, y2 = bbox
    length_px = abs(y2 - y1)
    width_px = abs(x2 - x1)

    area_px = 0
    if mask is not None:
        # Assuming mask is a numpy array of 0s and 1s
        area_px = np.sum(mask)
    else:
        area_px = length_px * width_px

    length_mm = length_px / PIXELS_PER_MM
    width_mm = width_px / PIXELS_PER_MM
    area_sq_mm = area_px / (PIXELS_PER_MM ** 2)

    WEIGHT_PER_SQ_MM = 6.67
    estimated_weight_mg = area_sq_mm * WEIGHT_PER_SQ_MM

    return length_mm, width_mm, area_sq_mm, estimated_weight_mg

# --- Upload Function ---
def upload_image_to_server(image_path, tray_number, count, avg_length, avg_weight, bounding_boxes, masks):
    """
    Uploads a processed image and its data to the Flask server.
    
    Args:
        image_path (str): The local path to the image file to be uploaded.
        tray_number (int): The detected tray number.
        count (int): The number of larvae detected.
        avg_length (float): The average length of larvae.
        avg_weight (float): The average weight of larvae.
        bounding_boxes (list): List of detected bounding boxes.
        masks (list): List of detected masks (converted to lists).
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        payload = {
            "image_data": encoded_string,
            "tray_number": tray_number,
            "count": count,
            "avg_length": avg_length,
            "avg_weight": avg_weight,
            "bounding_boxes": json.dumps(bounding_boxes),
            "masks": json.dumps(masks),
        }

        response = requests.post(f"{FLASK_SERVER_URL}/api/upload", json=payload)
        
        if response.status_code == 200:
            print(f"Successfully uploaded image {os.path.basename(image_path)} to server.")
        else:
            print(f"Failed to upload image. Server returned status code: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"An error occurred during image upload: {e}")
        traceback.print_exc()

# --- Main Processing Loop ---
def process_images_from_folder():
    """
    Monitors the input directory for new images, processes them,
    and publishes aggregated data to MQTT and the Flask server.
    """
    os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DETECTION_DIR, exist_ok=True)

    print(f"\n--- Checking for new images in {INPUT_IMAGE_DIR} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    images_found = False
    for filename in os.listdir(INPUT_IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images_found = True
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            print(f"Processing image: {image_path}")

            tray_number_str, ocr_confidence = extract_text_with_easyocr(image_path)
            if tray_number_str:
                try:
                    tray_number = int(tray_number_str)
                    print(f"Detected Tray Number: {tray_number} (Confidence: {ocr_confidence:.2f}%)")
                except ValueError:
                    print(f"Warning: Could not convert '{tray_number_str}' to an integer for tray number. Skipping larvae analysis for this image.")
                    tray_number = None
            else:
                print("No tray number detected by EasyOCR. Skipping larvae analysis for this image.")
                tray_number = None

            if tray_number is None:
                destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
                os.rename(image_path, destination_path)
                print(f"Moved image (no tray number detected): {image_path} to {destination_path}")
                continue

            print(f"Running Flat-Bug inference on {image_path}...")
            larvae_data_to_send = []
            total_count = 0
            
            try:
                # IMPORTANT FIX: The original code had two separate prediction blocks.
                # This has been consolidated into a single, correct block.
                prediction_results = flatbug_predictor.pyramid_predictions(
                    image_path,
                    scale_increment=2/3,
                    scale_before=1.0,
                    single_scale=False
                )

                if prediction_results and hasattr(prediction_results, 'boxes') and len(prediction_results.boxes) > 0:
                    total_count = len(prediction_results.boxes)
                    print(f"Found {total_count} larvae in Tray {tray_number}.")

                    # Create the output image with detections and save it
                    output_overview_path = os.path.join(OUTPUT_DETECTION_DIR, filename)
                    prediction_results.plot(
                        outpath=output_overview_path,
                        masks=True,
                        boxes=True,
                        confidence=True,
                        linewidth=2,
                        contour_color=(0, 255, 0),
                        box_color=(255, 0, 0)
                    )
                    print(f"Saved image with detections to: {output_overview_path}")

                    # Aggregate all bounding boxes and masks for the payload
                    bounding_boxes_payload = [box.tolist() for box in prediction_results.boxes]
                    masks_payload = [mask.data.cpu().numpy().tolist() for mask in prediction_results.masks]

                    # Calculate individual larva metrics and store them
                    for larva_id in range(total_count):
                        bbox_xyxy = prediction_results.boxes[larva_id].tolist()
                        larva_confidence = prediction_results.confs[larva_id].item()
                        mask_data = prediction_results.masks[larva_id].data.cpu().numpy().astype(np.uint8)

                        length_mm, width_mm, area_sq_mm, estimated_weight_mg = \
                            calculate_larva_metrics(bbox_xyxy, mask_data)

                        larvae_data_to_send.append({
                            "length": length_mm,
                            "weight": estimated_weight_mg
                        })
                        print(f"  Larva {larva_id + 1}: L={length_mm:.2f}mm, Wt={estimated_weight_mg:.2f}mg (Conf: {larva_confidence:.2f}%)")
                    
                    # Calculate aggregate metrics for the entire tray
                    avg_length = sum(d['length'] for d in larvae_data_to_send) / total_count
                    avg_weight = sum(d['weight'] for d in larvae_data_to_send) / total_count

                    # Publish aggregated data to MQTT
                    mqtt_payload = {
                        "tray_number": tray_number,
                        "avg_length": round(avg_length, 2),
                        "avg_weight": round(avg_weight, 2),
                        "count": total_count,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    print(f"Published data for Tray {tray_number} to MQTT.")
                    mqtt_client.publish(MQTT_TOPIC, json.dumps(mqtt_payload), qos=1)

                    # Upload the detected image and data to the Flask server
                    upload_image_to_server(
                        image_path=output_overview_path, # Use the image with detections
                        tray_number=tray_number,
                        count=total_count,
                        avg_length=round(avg_length, 2),
                        avg_weight=round(avg_weight, 2),
                        bounding_boxes=bounding_boxes_payload,
                        masks=masks_payload
                    )
                else:
                    print(f"No larvae detected by Flat-Bug in Tray {tray_number}.")

            except Exception as e:
                print(f"Error during Flat-Bug inference or data aggregation for {image_path}: {e}")
                traceback.print_exc()

            # Move the original image to the processed directory
            destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
            os.rename(image_path, destination_path)
            print(f"Moved processed image: {image_path} to {destination_path}")

    if not images_found:
        print("No new images found in the input folder.")

# --- Main Execution Block ---
if __name__ == "__main__":
    try:
        while True:
            process_images_from_folder()
            print(f"\nWaiting for {PROCESS_INTERVAL_SECONDS} seconds before checking again...")
            time.sleep(PROCESS_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nExiting program due to user interruption.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("Program finished.")
