import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt # Import MQTT library
import json # To send data as JSON
from math import sqrt
from flat_bug.predictor import Predictor
from flat_bug.config import DEFAULT_CFG, read_cfg # For configuration if needed
from flat_bug import logger as flatbug_logger, set_log_level # For flat-bug's internal logging

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883 # Standard unencrypted MQTT port
MQTT_TOPIC = "bsf_monitor/larvae_data" # <--- IMPORTANT: Make this topic unique for your project!
                                      # E.g., "your_username/bsf_monitor/larvae_data"

# --- Callbacks for MQTT Client ---
def on_connect(client, userdata, flags, rc, properties):
    """Callback function for when the MQTT client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

# --- Configuration ---
INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img" # <--- IMPORTANT: SET YOUR INPUT IMAGE FOLDER HERE!
PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/processed_images" # Directory to move processed images. Sort and change images here after processing.
OUTPUT_DETECTION_DIR = "/home/pato/Documents/sdf/BSF-pi-script/detected_images" # Directory for saving images with detections

# Ensure output directories exist
os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DETECTION_DIR, exist_ok=True)

# EasyOCR Settings
EASYOCR_LANGUAGES = ['en'] # Languages to load. 'en' for English.
EASYOCR_ALLOWLIST = '0123456789'
EASYOCR_BLOCKLIST = ''

# Script Timing
PROCESS_INTERVAL_SECONDS = 10 # How often to check for new images and process them

# --- Model Initialization ---
# Assuming you have the flat_bug_M.pt model file in your project directory
MODEL_PATH = "/home/pato/Documents/sdf/bestmodel.onnx"
# The default config in the flat-bug library is suitable for our needs
cfg = read_cfg(DEFAULT_CFG)
try:
    print(f"Initializing Flat-Bug model from {MODEL_PATH}...")
    flatbug_predictor = Predictor.from_path(MODEL_PATH)
    print("Flat-Bug model initialized successfully.")
    # Set the logging level for flat-bug's internal logger
    set_log_level("INFO")
except Exception as e:
    print(f"Error initializing Flat-Bug model: {e}")
    flatbug_predictor = None

# Initialize EasyOCR reader once to save time
try:
    print("Initializing EasyOCR reader...")
    ocr_reader = easyocr.Reader(EASYOCR_LANGUAGES)
    print("EasyOCR reader initialized.")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    ocr_reader = None

# --- MQTT Client Setup ---
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start() # Start the loop in a separate thread
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")

# --- Helper Functions ---
def get_tray_number(image_path):
    """
    Extracts the tray number from the image using OCR.
    Assumes the tray number is the first detected number in the image.
    """
    if not ocr_reader:
        return None
    try:
        # Load the image for OCR
        img_ocr = cv2.imread(image_path)
        # Use EasyOCR to read text from the image
        results = ocr_reader.readtext(img_ocr, allowlist=EASYOCR_ALLOWLIST)
        
        # We assume the first detected number is the tray number
        if results:
            for (bbox, text, prob) in results:
                # Basic check to ensure it looks like a number
                if text.isdigit() and len(text) <= 5: 
                    return int(text)
        return None
    except Exception as e:
        print(f"Error performing OCR on image {image_path}: {e}")
        return None

def calculate_larvae_metrics(predictions):
    """
    Calculates average length, width, area, and weight from a list of predictions.
    This is a simplified example. You would need to calibrate these metrics.
    """
    if not predictions:
        return 0, 0, 0, 0
    
    larvae_data = []
    for pred in predictions:
        # Using the bounding box to approximate metrics
        x1, y1, x2, y2 = pred.box.xyxy[0].cpu().numpy()
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Simplified metrics based on bounding box
        length = max(width, height)
        avg_width = min(width, height)
        
        larvae_data.append({
            'length': length,
            'width': avg_width,
            'area': area,
            'weight': area * 0.05 # Placeholder conversion factor
        })
    
    if not larvae_data:
        return 0, 0, 0, 0
    
    avg_length = np.mean([d['length'] for d in larvae_data])
    avg_width = np.mean([d['width'] for d in larvae_data])
    avg_area = np.mean([d['area'] for d in larvae_data])
    avg_weight = np.mean([d['weight'] for d in larvae_data])
    
    return round(avg_length, 2), round(avg_width, 2), round(avg_area, 2), round(avg_weight, 2)

def process_images_from_folder():
    """
    Scans the input directory for new images, processes them, and moves them.
    """
    if not flatbug_predictor:
        print("Flat-Bug predictor is not initialized. Exiting.")
        return

    images_found = False
    print(f"Checking for new images in {INPUT_IMAGE_DIR}...")
    for filename in sorted(os.listdir(INPUT_IMAGE_DIR)):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(INPUT_IMAGE_DIR, filename)
        images_found = True

        try:
            # Step 1: Read the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image at {image_path}. Skipping.")
                continue

            # Step 2: Get the tray number using OCR
            tray_number = get_tray_number(image_path)
            if tray_number is None:
                print(f"Warning: Could not detect a tray number for image {image_path}. Skipping.")
                continue
            
            # Step 3: Run inference with Flat-Bug
            flatbug_logger.info(f"Processing image for larvae detection: {image_path}")
            
            # We are now using the `predict` method, which is more robust for a single image.
            # The original `pyramid_predictions` was likely causing the batching error.
            # The `predict` function expects a single image and handles the necessary transformations internally.
            
            prediction_results = flatbug_predictor.predict(
                image=image_path
            )

            # Check if any larvae were detected
            if prediction_results and prediction_results.pred:
                larvae_count = len(prediction_results.pred[0])
                avg_length, avg_width, avg_area, avg_weight = calculate_larvae_metrics(prediction_results.pred[0])
                print(f"Detected {larvae_count} larvae for Tray {tray_number}.")

                # Save the image with detections for visual inspection
                annotated_image = flatbug_predictor.annotate(prediction_results)
                output_path = os.path.join(OUTPUT_DETECTION_DIR, f"detected_{filename}")
                cv2.imwrite(output_path, annotated_image)
                print(f"Saved image with detections to {output_path}")

                # Step 4: Aggregate and send data
                timestamp = datetime.now().isoformat()
                payload = {
                    "tray_number": tray_number,
                    "image_path": image_path,
                    "timestamp": timestamp,
                    "larvae_count": larvae_count,
                    "avg_length": avg_length,
                    "avg_width": avg_width,
                    "avg_area": avg_area,
                    "avg_weight": avg_weight
                }

                # Publish data to MQTT
                try:
                    mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
                    print(f"Data published successfully to MQTT broker.")
                except Exception as mqtt_e:
                    print(f"Error publishing data to MQTT broker: {mqtt_e}")
            else:
                print(f"No larvae detected for Tray {tray_number}. No data published to MQTT.")

        except Exception as e:
            print(f"Error during Flat-Bug inference or data aggregation for {image_path}: {e}")
            import traceback
            traceback.print_exc()

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
        import traceback
        traceback.print_exc()
    finally:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("Program finished.")
