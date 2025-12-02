# import time
# import cv2
# import os
# import easyocr
# from datetime import datetime
# import numpy as np
# import paho.mqtt.client as mqtt # Import MQTT library
# import json # To send data as JSON
# import requests
# from camera_capture import capture_image

# # --- Flat-Bug Model Imports ---
# from flat_bug.predictor import Predictor
# from flat_bug.config import DEFAULT_CFG, read_cfg # For configuration if needed
# from flat_bug import logger as flatbug_logger, set_log_level # For flat-bug's internal logging

# # --- Web API Configuration ---
# # Set the URL for your web application's API endpoint
# WEB_APP_API_URL = "https://soldierfly-fly-monitor.onrender.com/api/larvae_data" # <--- IMPORTANT: CHANGE TO YOUR WEB APP'S API URL
# #B_APP_API_URL = "http://192.168.162.253:8000/api/larvae_data" # Local testing URL

# # --- MQTT Configuration ---
# MQTT_BROKER = "broker.hivemq.com"
# MQTT_PORT = 1883 # Standard unencrypted MQTT port
# MQTT_TOPIC = "bsf_monitor/larvae_data" # <--- IMPORTANT: Make this topic unique for your project!
#                                       # E.g., "your_username/bsf_monitor/larvae_data"

# # --- Configuration ---
# INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img" # <--- IMPORTANT: SET YOUR INPUT IMAGE FOLDER HERE!
# PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/processed_images" # Directory to move processed images. Sort and change images here after processing.
# OUTPUT_DETECTION_DIR = "/home/pato/Documents/sdf/BSF-pi-script/detected_images" # Images with bounding boxes

# # EasyOCR Settings
# EASYOCR_LANGUAGES = ['en'] # Languages to load. 'en' for English.
# EASYOCR_ALLOWLIST = '0123456789' # Only allow digits for tray number recognition
# EASYOCR_BLOCKLIST = ''

# # Script Timing
# PROCESS_INTERVAL_SECONDS = 20 # How often to check for new images and process them

# # Flat-Bug Model Configuration
# FLATBUG_MODEL_PATH = "/home/pato/Documents/sdf/YoloRetrain.pt" # <--- IMPORTANT: SET PATH TO YOUR DOWNLOADED FLAT-BUG MODEL WEIGHTS (.pt file)
# FLATBUG_DEVICE = "cpu" # Recommended for Raspberry Pi or systems without dedicated GPU
# FLATBUG_DTYPE = "float32" # Use float32 for CPU, float16 for GPU if supported

# # Calibration Factor (pixels per millimeter)
# PIXELS_PER_MM = 20.0

# # =============================================================================
# # === STEP 2: INITIALIZE EASY OCR
# # =============================================================================
# def initialize_easyocr():
#     """Initializes and returns the EasyOCR reader object."""
#     print("Initializing EasyOCR reader. This may download models on first run...")
#     try:
#         reader = easyocr.Reader(EASYOCR_LANGUAGES, recog_network='latin_g2', gpu=False)
#         print("EasyOCR reader initialized successfully for integer-only recognition.")
#         return reader
#     except Exception as e:
#         print(f"Error initializing EasyOCR: {e}")
#         print("Please ensure you have an internet connection for the first run to download models.")
#         exit()

# # =============================================================================
# # === STEP 4: LOAD FLATBUG MODEL
# # =============================================================================
# def initialize_flatbug_model():
#     """Loads and returns the Flat-Bug predictor object."""
#     print(f"Loading Flat-Bug model from: {FLATBUG_MODEL_PATH} on device: {FLATBUG_DEVICE}...")
#     try:
#         flatbug_config = DEFAULT_CFG
#         # You can customize flatbug_config here, e.g., flatbug_config["SCORE_THRESHOLD"] = 0.6
#         flatbug_predictor = Predictor(
#             FLATBUG_MODEL_PATH,
#             device=FLATBUG_DEVICE,
#             dtype=FLATBUG_DTYPE,
#             cfg=flatbug_config
#         )
#         print("Flat-Bug model loaded successfully.")
#         return flatbug_predictor
#     except Exception as e:
#         print(f"Error loading Flat-Bug model: {e}")
#         print("Please ensure your model path is correct and flat-bug library is installed.")
#         exit()

# # =============================================================================
# # === STEP 7: CONNECT TO MQTT BROKER
# # =============================================================================
# def on_connect(client, userdata, flags, rc, properties):
#     """Callback function for when the MQTT client connects to the broker."""
#     if rc == 0:
#         print("Connected to MQTT Broker!")
#     else:
#         print(f"Failed to connect, return code {rc}\n")

# def initialize_mqtt_client():
#     """Initializes, connects, and returns the MQTT client object."""
#     mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
#     mqtt_client.on_connect = on_connect
#     try:
#         mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
#         mqtt_client.loop_start() 
#         return mqtt_client
#     except Exception as e:
#         print(f"Failed to connect to MQTT broker: {e}")
#         exit()

# # =============================================================================
# # === HELPER FUNCTIONS (PREPROCESSING, OCR, METRICS)
# # =============================================================================

# def preprocess_image_for_easyocr(image):
#     """Converts image to grayscale and applies median blur for OCR."""
#     if image is None or image.size == 0:
#         return None
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     denoised = cv2.medianBlur(gray, 3) 
#     return denoised

# # --- STEP 3: RETURN THE TRAY ID ---
# def get_tray_id(image_path, reader):
#     """
#     Extracts integer text (assumed to be tray number) from an image using EasyOCR.
#     Returns the extracted integer as a string and its confidence.
#     """
#     image = cv2.imread(image_path)
#     if image is None:
#         print(f"Error: Could not read image {image_path}")
#         return None, 0

#     processed_image = preprocess_image_for_easyocr(image)
#     if processed_image is None:
#         return None, 0

#     extracted_integers = []
#     all_confidences = []

#     try:
#         results = reader.readtext(processed_image, allowlist=EASYOCR_ALLOWLIST)

#         for (bbox, text, confidence) in results:
#             if text.strip():
#                 cleaned_text = ''.join(filter(str.isdigit, text.strip()))
#                 if cleaned_text:
#                     try:
#                         integer_value = int(cleaned_text)
#                         extracted_integers.append(str(integer_value))
#                         all_confidences.append(float(confidence))
#                     except ValueError:
#                         print(f"  Skipping non-integer segment after filtering: '{cleaned_text}' from original '{text}'")
#                 else:
#                     print(f"  No digits found after filtering: '{text}'")
#             else:
#                 print("  Skipping empty text detection.")

#     except Exception as e:
#         print(f"Error during EasyOCR processing: {e}")
#         return None, 0

#     if extracted_integers:
#         final_extracted_text = extracted_integers[0]
#         overall_avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
#         return final_extracted_text, overall_avg_confidence * 100
#     else:
#         return None, 0

# # --- STEP 5: DETECT THE LARVAE ---
# def detect_larvae(image_path, predictor):
#     """Runs the Flat-Bug model on the image and returns prediction results."""
#     print(f"Running Flat-Bug inference on {image_path}...")
#     try:
#         prediction_results = predictor.pyramid_predictions(
#             image_path,
#             scale_increment=2/3,
#             scale_before=1.0,
#             single_scale=False
#         )
#         return prediction_results
#     except Exception as e:
#         print(f"Error during Flat-Bug inference for {image_path}: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

# # --- STEP 6: COMPUTE THE WEIGHT, COUNT ---
# def calculate_larva_metrics(bbox, mask=None):
#     """
#     Calculates larva length, width, area, and estimated weight based on bounding box and mask.
#     Assumes bbox is [x1, y1, x2, y2] in pixels.
#     """
#     x1, y1, x2, y2 = bbox
#     length_px = abs(y2 - y1)
#     width_px = abs(x2 - x1)

#     area_px = 0
#     if mask is not None:
#         area_px = np.sum(mask)
#     else:
#         area_px = length_px * width_px

#     length_mm = length_px / PIXELS_PER_MM
#     width_mm = width_px / PIXELS_PER_MM
#     area_sq_mm = area_px / (PIXELS_PER_MM ** 2)

#     WEIGHT_PER_SQ_MM = 6.67
#     estimated_weight_mg = area_sq_mm * WEIGHT_PER_SQ_MM

#     return length_mm, width_mm, area_sq_mm, estimated_weight_mg

# def compute_and_aggregate_metrics(prediction_results, tray_number, filename):
#     """
#     Processes prediction results to compute metrics for each larva
#     and returns aggregated data payload.
#     """
#     larvae_data_to_send = []
#     total_count = 0

#     if prediction_results and hasattr(prediction_results, 'boxes') and prediction_results.boxes is not None and len(prediction_results.boxes) > 0:
#         total_count = len(prediction_results.boxes)
#         print(f"Found {total_count} larvae in Tray {tray_number}.")

#         # Use prediction_results.plot() for overview image
#         output_overview_path = os.path.join(OUTPUT_DETECTION_DIR, filename)
#         prediction_results.plot(
#             outpath=output_overview_path,
#             masks=True, # Set to True if your model predicts masks and you want to visualize them
#             boxes=True,
#             confidence=True,
#             linewidth=2,
#             contour_color=(0, 255, 0), # Green for mask contours
#             box_color=(255, 0, 0) # Red for bounding boxes
#         )
#         print(f"Saved image with detections to: {output_overview_path}")

#         # Continue with calculating metrics for MQTT payload
#         for larva_id in range(total_count):
#             bbox_xyxy = prediction_results.boxes[larva_id].tolist()
#             larva_confidence = prediction_results.confs[larva_id].item()

#             mask = None
#             if hasattr(prediction_results, 'masks') and prediction_results.masks is not None and len(prediction_results.masks) > larva_id:
#                 larva_mask_object = prediction_results.masks[larva_id]
#                 mask = larva_mask_object.data.cpu().numpy().astype(np.uint8)

#             length_mm, width_mm, area_sq_mm, estimated_weight_mg = \
#                 calculate_larva_metrics(bbox_xyxy, mask)

#             larvae_data_to_send.append({
#                 "tray_number": tray_number,
#                 "length": round(length_mm, 2),
#                 "width": round(width_mm, 2),
#                 "area": round(area_sq_mm, 2),
#                 "weight": round(estimated_weight_mg, 2),
#                 "count": 1
#             })
#             print(f"  Larva {larva_id + 1}: L={length_mm:.2f}mm, W={width_mm:.2f}mm, A={area_sq_mm:.2f}mm², Wt={estimated_weight_mg:.2f}mg (Conf: {larva_confidence:.2f}%)")
    
#     else:
#         print(f"No larvae detected by Flat-Bug in Tray {tray_number}.")
#         return None, 0 # Return None payload and 0 count

#     if total_count > 0:
#         avg_length = sum(d['length'] for d in larvae_data_to_send) / total_count
#         avg_width = sum(d['width'] for d in larvae_data_to_send) / total_count
#         avg_area = sum(d['area'] for d in larvae_data_to_send) / total_count
#         avg_weight = sum(d['weight'] for d in larvae_data_to_send) / total_count

#         payload = {
#             "tray_number": tray_number,
#             "length": round(avg_length, 2),
#             "width": round(avg_width, 2),
#             "area": round(avg_area, 2),
#             "weight": round(avg_weight, 2),
#             "count": total_count
#         }
#         return payload, total_count
    
#     return None, 0

# # --- STEP 8: UPLOAD THE ID, LARVAE DETAILS TO THE WEB (AND MQTT) ---
# def publish_data(mqtt_client, payload):
#     """Publishes the data payload to the MQTT broker and/or web API."""
#     print(f"Publishing aggregated data for Tray {payload['tray_number']}...")
#     try:
#         mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
#         print(f"Data published successfully to MQTT broker on topic '{MQTT_TOPIC}'.")
        
#     except requests.exceptions.RequestException as api_e:
#         print(f"Error uploading data to web API: {api_e}")
#     except Exception as mqtt_e:
#         print(f"Error publishing data to MQTT broker: {mqtt_e}")

# # =============================================================================
# # === MAIN PROCESSING FUNCTION
# # =============================================================================

# def process_available_images(reader, predictor, mqtt_client):
#     """
#     Monitors the input directory for new images, processes them,
#     and publishes aggregated data.
#     Returns True if any images were found and processed, False otherwise.
#     """
#     print(f"\n--- Checking for new images in {INPUT_IMAGE_DIR} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

#     images_found = False
#     for filename in os.listdir(INPUT_IMAGE_DIR):
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#             images_found = True
#             image_path = os.path.join(INPUT_IMAGE_DIR, filename)
#             print(f"Processing image: {image_path}")

#             # --- STEP 3: Return the tray ID ---
#             tray_number_str, ocr_confidence = get_tray_id(image_path, reader)
            
#             if tray_number_str:
#                 try:
#                     tray_number = int(tray_number_str)
#                     print(f"Detected Tray Number: {tray_number} (Confidence: {ocr_confidence:.2f}%)")
#                 except ValueError:
#                     print(f"Warning: Could not convert '{tray_number_str}' to an integer for tray number. Skipping larvae analysis for this image.")
#                     tray_number = None
#             else:
#                 print("No tray number detected by EasyOCR. Skipping larvae analysis for this image.")
#                 tray_number = None

#             if tray_number is None:
#                 destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
#                 os.rename(image_path, destination_path)
#                 print(f"Moved image (no tray number detected): {image_path} to {destination_path}")
#                 continue # Move to the next image

#             # --- STEP 5: Detect the larvae ---
#             prediction_results = detect_larvae(image_path, predictor)

#             # --- STEP 6: Compute the weight, count ---
#             payload, total_count = compute_and_aggregate_metrics(prediction_results, tray_number, filename)

#             # --- STEP 8: Upload the ID, larvae details ---
#             if total_count > 0 and payload is not None:
#                 publish_data(mqtt_client, payload)
#             else:
#                 print(f"No data to publish for Tray {tray_number} (no larvae detected).")

#             # Move the processed image
#             destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
#             os.rename(image_path, destination_path)
#             print(f"Moved processed image: {image_path} to {destination_path}")

#     if not images_found:
#         print("No new images found in the input folder.")
        
#     return images_found

# # =============================================================================
# # === MAIN EXECUTION BLOCK
# # =============================================================================

# def main():
#     """
#     Main function to initialize models and run the processing loop.
#     """
#     # Create directories if they don't exist
#     os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
#     os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
#     os.makedirs(OUTPUT_DETECTION_DIR, exist_ok=True)

#     # --- Initialize components ONCE ---
#     # STEP 2: Initialize EasyOCR
#     reader = initialize_easyocr()
#     # STEP 4: Load Flatbug
#     predictor = initialize_flatbug_model()
#     # STEP 7: Connect to MQTT Broker
#     mqtt_client = initialize_mqtt_client()

#     try:
#         while True:
#             # --- STEP 1: Check the folder for image available ---
#             images_were_processed = process_available_images(reader, predictor, mqtt_client)
            
#             # --- (Part of STEP 1): Initialize camera (capture image if none found) ---
#             if not images_were_processed:
#                 print("No images to process. Capturing a new one.")
#                 capture_image(INPUT_IMAGE_DIR)
#                 # After capturing, we can either wait for the next loop or process immediately
#                 # Let's process immediately to reduce delay
#                 print("Processing newly captured image...")
#                 process_available_images(reader, predictor, mqtt_client)


#             print(f"\nWaiting for {PROCESS_INTERVAL_SECONDS} seconds before checking again...")
#             time.sleep(PROCESS_INTERVAL_SECONDS)

#     except KeyboardInterrupt:
#         print("\nExiting program due to user interruption.")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         import traceback
#         traceback.print_exc()
#     finally:
#         mqtt_client.loop_stop()
#         mqtt_client.disconnect()
#         print("MQTT client disconnected.")
#         print("Program finished.")


# if __name__ == "__main__":
#     main()




import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt
import json
import requests
import base64
from camera_capture import capture_image

# --- Flat-Bug Model Imports ---
from flat_bug.predictor import Predictor
from flat_bug.config import DEFAULT_CFG, read_cfg
from flat_bug import logger as flatbug_logger, set_log_level

# --- Web API Configuration ---
# ADDED: Direct API endpoint for image uploads
WEB_APP_API_URL = "https://bsf-larvae-monitoring.onrender.com/api/upload"  # CHANGED: Use upload endpoint
# WEB_APP_API_URL = "http://localhost:8000/api/upload"  # For local testing

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "bsf_monitor/larvae_data"

# --- Configuration ---
INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img"
PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/processed_images"
OUTPUT_DETECTION_DIR = "/home/pato/Documents/sdf/BSF-pi-script/detected_images"

# EasyOCR Settings
EASYOCR_LANGUAGES = ['en']
EASYOCR_ALLOWLIST = '0123456789'
EASYOCR_BLOCKLIST = ''

# Script Timing
PROCESS_INTERVAL_SECONDS = 20

# Flat-Bug Model Configuration
FLATBUG_MODEL_PATH = "/home/pato/Documents/sdf/YoloRetrain.pt"
FLATBUG_DEVICE = "cpu"
FLATBUG_DTYPE = "float32"

# Calibration Factor (pixels per millimeter)
PIXELS_PER_MM = 20.0

# =============================================================================
# === STEP 2: INITIALIZE EASY OCR
# =============================================================================
def initialize_easyocr():
    """Initializes and returns the EasyOCR reader object."""
    print("Initializing EasyOCR reader. This may download models on first run...")
    try:
        reader = easyocr.Reader(EASYOCR_LANGUAGES, recog_network='latin_g2', gpu=False)
        print("EasyOCR reader initialized successfully for integer-only recognition.")
        return reader
    except Exception as e:
        print(f"Error initializing EasyOCR: {e}")
        print("Please ensure you have an internet connection for the first run to download models.")
        exit()

# =============================================================================
# === STEP 4: LOAD FLATBUG MODEL
# =============================================================================
def initialize_flatbug_model():
    """Loads and returns the Flat-Bug predictor object."""
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
        return flatbug_predictor
    except Exception as e:
        print(f"Error loading Flat-Bug model: {e}")
        print("Please ensure your model path is correct and flat-bug library is installed.")
        exit()

# =============================================================================
# === STEP 7: CONNECT TO MQTT BROKER
# =============================================================================
def on_connect(client, userdata, flags, rc, properties):
    """Callback function for when the MQTT client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

def initialize_mqtt_client():
    """Initializes, connects, and returns the MQTT client object."""
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start() 
        return mqtt_client
    except Exception as e:
        print(f"Failed to connect to MQTT broker: {e}")
        exit()

# =============================================================================
# === HELPER FUNCTIONS
# =============================================================================

def preprocess_image_for_easyocr(image):
    """Converts image to grayscale and applies median blur for OCR."""
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3) 
    return denoised

def get_tray_id(image_path, reader):
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

def detect_larvae(image_path, predictor):
    """Runs the Flat-Bug model on the image and returns prediction results."""
    print(f"Running Flat-Bug inference on {image_path}...")
    try:
        prediction_results = predictor.pyramid_predictions(
            image_path,
            scale_increment=2/3,
            scale_before=1.0,
            single_scale=False
        )
        return prediction_results
    except Exception as e:
        print(f"Error during Flat-Bug inference for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        area_px = np.sum(mask)
    else:
        area_px = length_px * width_px

    length_mm = length_px / PIXELS_PER_MM
    width_mm = width_px / PIXELS_PER_MM
    area_sq_mm = area_px / (PIXELS_PER_MM ** 2)

    WEIGHT_PER_SQ_MM = 6.67
    estimated_weight_mg = area_sq_mm * WEIGHT_PER_SQ_MM

    return length_mm, width_mm, area_sq_mm, estimated_weight_mg

# CHANGED: Simplified to only compute metrics (no image data)
def compute_and_aggregate_metrics(prediction_results, tray_number):
    """
    Processes prediction results to compute metrics for each larva.
    Returns only the metrics payload for MQTT.
    """
    larvae_data_to_send = []
    total_count = 0

    if prediction_results and hasattr(prediction_results, 'boxes') and prediction_results.boxes is not None and len(prediction_results.boxes) > 0:
        total_count = len(prediction_results.boxes)
        print(f"Found {total_count} larvae in Tray {tray_number}.")

        # Save detection image locally (optional)
        output_overview_path = os.path.join(OUTPUT_DETECTION_DIR, f"detected_{tray_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
        prediction_results.plot(
            outpath=output_overview_path,
            masks=True,
            boxes=True,
            confidence=True,
            linewidth=2,
            contour_color=(0, 255, 0),
            box_color=(255, 0, 0)
        )
        print(f"Saved detection image to: {output_overview_path}")

        # Calculate metrics
        for larva_id in range(total_count):
            bbox_xyxy = prediction_results.boxes[larva_id].tolist()
            larva_confidence = prediction_results.confs[larva_id].item()

            mask = None
            if hasattr(prediction_results, 'masks') and prediction_results.masks is not None and len(prediction_results.masks) > larva_id:
                larva_mask_object = prediction_results.masks[larva_id]
                mask = larva_mask_object.data.cpu().numpy().astype(np.uint8)

            length_mm, width_mm, area_sq_mm, estimated_weight_mg = calculate_larva_metrics(bbox_xyxy, mask)

            larvae_data_to_send.append({
                "length": round(length_mm, 2),
                "width": round(width_mm, 2),
                "area": round(area_sq_mm, 2),
                "weight": round(estimated_weight_mg, 2),
                "count": 1
            })
            print(f"  Larva {larva_id + 1}: L={length_mm:.2f}mm, W={width_mm:.2f}mm, A={area_sq_mm:.2f}mm², Wt={estimated_weight_mg:.2f}mg")

    else:
        print(f"No larvae detected by Flat-Bug in Tray {tray_number}.")
        total_count = 0

    # Create metrics payload for MQTT
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
            "count": total_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        return payload, total_count
    else:
        # Return empty payload if no larvae detected
        payload = {
            "tray_number": tray_number,
            "length": 0,
            "width": 0,
            "area": 0,
            "weight": 0,
            "count": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        return payload, 0

# # NEW: Separate function for image upload via API
# def upload_image_to_api(image_path, tray_number, count, avg_length, avg_weight, bounding_boxes, masks):
#     """Uploads image and detection data directly to web app API."""
#     try:
#         # Read and encode image
#         with open(image_path, "rb") as image_file:
#             image_data_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
#         # Prepare payload for API
#         api_payload = {
#             'image_data': image_data_base64,
#             'tray_number': tray_number,
#             'count': count,
#             'avg_length': avg_length,
#             'avg_weight': avg_weight,
#             'bounding_boxes': json.dumps(bounding_boxes) if bounding_boxes else "[]",
#             'masks': json.dumps(masks) if masks else "[]"
#         }
        
#         print(f"Uploading image for Tray {tray_number} to API...")
        
#         # Send to web app API
#         response = requests.post(WEB_APP_API_URL, json=api_payload, timeout=30)
        
#         if response.status_code == 200:
#             print(f"✅ Image uploaded successfully via API for Tray {tray_number}")
#             return True
#         else:
#             print(f"❌ API upload failed: {response.status_code} - {response.text}")
#             return False
            
#     except Exception as e:
#         print(f"❌ Error uploading image via API: {e}")
#         return False


def upload_image_to_api(image_path, tray_number, count, avg_length, avg_weight, bounding_boxes, masks):
    """Uploads image and detection data directly to web app API."""
    try:
        # Read and compress image before encoding - PRESERVES ORIGINAL COLORS
        print(f"Reading and compressing image for Tray {tray_number}...")
        
        # Read image with OpenCV - this maintains original colors
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return False
        
        # Convert from BGR to RGB to maintain correct color representation
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if too large (same as your main app)
        max_dimension = 1200
        height, width = img_rgb.shape[:2]
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            print(f"  Resized image from {width}x{height} to {new_width}x{new_height}")
        
        # Compress as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, encoded_img = cv2.imencode('.jpg', img_rgb, encode_params)
        
        if not success:
            print("❌ Failed to encode image")
            return None
            
        # Convert to base64
        image_data_base64 = base64.b64encode(encoded_img.tobytes()).decode('utf-8')
        file_size_kb = len(image_data_base64) / 1024
        
        print(f"  Compressed size: {file_size_kb:.1f} KB")

        # TEST MODE: Send empty detection data
        print("🔄 TEST MODE: Sending empty detection data (bounding_boxes=[], masks=[])")
        
        # Prepare payload for API
        api_payload = {
            'image_data': image_data_base64,
            'tray_number': tray_number,
            'count': count,
            'avg_length': avg_length,
            'avg_weight': avg_weight,
            'bounding_boxes': json.dumps(bounding_boxes) if bounding_boxes else "[]",
            'masks': json.dumps(masks) if masks else "[]"
        }
        
        print(f"Uploading image for Tray {tray_number} to API...")
        
        # Use simple requests.post (NO Session) - This is what works in test script
        response = requests.post(
            WEB_APP_API_URL,
            json=api_payload,
            timeout=120,  # 2 minute timeout
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"✅ SUCCESS: {result.get('message', 'Upload successful')}")
                return True
            except json.JSONDecodeError as e:
                print(f"⚠️  Server returned 200 but invalid JSON: {e}")
                # Even if JSON fails, status 200 means upload likely worked
                print("✅ Upload likely successful despite JSON issue")
                return True
        else:
            print(f"❌ API upload failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Upload took too long (over 120 seconds)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ CONNECTION ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False



def test_api_connection():
    """Test basic connection to the API"""
    print("🔍 Testing API connection...")
    try:
        # Test if the main site is reachable
        response = requests.get("https://bsf-larvae-monitoring.onrender.com", timeout=10)
        if response.status_code == 200:
            print("✅ Main site is reachable")
            return True
        else:
            print(f"❌ Main site returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot reach main site: {e}")
        return False



# NEW: Retry logic for image upload
def upload_image_to_api_with_retry(image_path, tray_number, count, avg_length, avg_weight, bounding_boxes, masks, max_retries=2):
    """Upload image with retry logic for transient failures."""
    for attempt in range(max_retries + 1):
        print(f"📤 Upload attempt {attempt + 1}/{max_retries + 1} for Tray {tray_number}")
        success = upload_image_to_api(image_path, tray_number, count, avg_length, avg_weight, bounding_boxes, masks)
        if success:
            return True
        elif attempt < max_retries:
            wait_time = (attempt + 1) * 5  # 5, 10 seconds between retries
            print(f"🔄 Retry {attempt + 1}/{max_retries} in {wait_time} seconds...")
            time.sleep(wait_time)
        else:
            print(f"💥 All upload attempts failed for Tray {tray_number}")
    
    return False

# NEW: Extract bounding boxes and masks with limit
def extract_detection_data(prediction_results, max_detections=50):
    """Extracts bounding boxes and masks from prediction results for API upload, with limit to avoid oversized payloads."""
    bounding_boxes = []
    masks = []
    
    if prediction_results and hasattr(prediction_results, 'boxes') and prediction_results.boxes is not None:
        total_detected = len(prediction_results.boxes)
        # Limit the number of detections we send to avoid oversized payloads
        limited_count = min(total_detected, max_detections)
        
        print(f"🔍 Limiting API data: sending {limited_count} detections (of {total_detected} total)")
        
        for larva_id in range(limited_count):
            bbox_xyxy = prediction_results.boxes[larva_id].tolist()
            bounding_boxes.append(bbox_xyxy)
            
            if hasattr(prediction_results, 'masks') and prediction_results.masks is not None and len(prediction_results.masks) > larva_id:
                larva_mask_object = prediction_results.masks[larva_id]
                mask = larva_mask_object.data.cpu().numpy().astype(np.uint8)
                masks.append(mask.tolist())
    
    return bounding_boxes, masks

def publish_data(mqtt_client, payload):
    """Publishes the metrics data payload to the MQTT broker."""
    print(f"Publishing metrics data for Tray {payload['tray_number']} via MQTT...")
    try:
        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
        print(f"✅ Metrics data published to MQTT for Tray {payload['tray_number']}")
        
    except Exception as mqtt_e:
        print(f"❌ Error publishing data to MQTT broker: {mqtt_e}")

# =============================================================================
# === MAIN PROCESSING FUNCTION
# =============================================================================

def process_available_images(reader, predictor, mqtt_client):
    """
    Monitors the input directory for new images, processes them,
    and publishes data via both MQTT (metrics) and API (images).
    """
    print(f"\n--- Checking for new images in {INPUT_IMAGE_DIR} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    images_found = False
    for filename in os.listdir(INPUT_IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images_found = True
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            print(f"Processing image: {image_path}")

            # Extract tray number
            tray_number_str, ocr_confidence = get_tray_id(image_path, reader)
            
            if tray_number_str:
                try:
                    tray_number = int(tray_number_str)
                    print(f"Detected Tray Number: {tray_number} (Confidence: {ocr_confidence:.2f}%)")
                except ValueError:
                    print(f"Warning: Could not convert '{tray_number_str}' to an integer. Skipping.")
                    tray_number = None
            else:
                print("No tray number detected by EasyOCR. Skipping.")
                tray_number = None

            if tray_number is None:
                # Move image even if no tray number detected
                destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
                os.rename(image_path, destination_path)
                print(f"Moved image (no tray number): {image_path} to {destination_path}")
                continue

            # Detect larvae
            prediction_results = detect_larvae(image_path, predictor)

            # CHANGED: Get metrics for MQTT
            metrics_payload, total_count = compute_and_aggregate_metrics(prediction_results, tray_number)

            # NEW: Extract detection data for API
            bounding_boxes, masks = extract_detection_data(prediction_results)

            # ADD THIS DEBUG LINE:
            print(f"🔍 DEBUG: Extracted {len(bounding_boxes)} bounding boxes, {len(masks)} masks")

            # Send metrics via MQTT
            publish_data(mqtt_client, metrics_payload)

            # NEW: Upload image via API (only if we have detection results)
            if prediction_results is not None:
                upload_success = upload_image_to_api_with_retry(
                    image_path=image_path,
                    tray_number=tray_number,
                    count=total_count,
                    avg_length=metrics_payload['length'],
                    avg_weight=metrics_payload['weight'],
                    bounding_boxes=bounding_boxes,
                    masks=masks,
                    max_retries=2  # Will try 3 times total (original + 2 retries)
                )
                
                if not upload_success:
                    print(f"⚠️  Image upload failed for Tray {tray_number}, but metrics were sent via MQTT")

            # Move the processed image
            destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
            os.rename(image_path, destination_path)
            print(f"Moved processed image: {image_path} to {destination_path}")

    if not images_found:
        print("No new images found in the input folder.")
        
    return images_found

# =============================================================================
# === MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    Main function to initialize models and run the processing loop.
    """
    # Create directories if they don't exist
    os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DETECTION_DIR, exist_ok=True)

    # Initialize components
    reader = initialize_easyocr()
    predictor = initialize_flatbug_model()
    mqtt_client = initialize_mqtt_client()

    # TEST CONNECTION FIRST
    print("=== CONNECTION TEST ===")
    test_api_connection()
    print("=======================")

    try:
        while True:
            # Check for new images
            images_were_processed = process_available_images(reader, predictor, mqtt_client)
            
            # Capture new image if none found
            if not images_were_processed:
                print("No images to process. Capturing a new one.")
                capture_image(INPUT_IMAGE_DIR)
                print("Processing newly captured image...")
                process_available_images(reader, predictor, mqtt_client)

            print(f"\nWaiting for {PROCESS_INTERVAL_SECONDS} seconds before checking again...")
            time.sleep(PROCESS_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nExiting program due to user interruption.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("MQTT client disconnected.")
        print("Program finished.")


if __name__ == "__main__":
    main()