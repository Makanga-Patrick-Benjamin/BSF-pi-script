import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt # Import MQTT library
import json # To send data as JSON
import traceback # To print full traceback for debugging

# --- Flat-Bug Model Imports ---
from flat_bug.predictor import Predictor
from flat_bug.config import DEFAULT_CFG, read_cfg # For configuration if needed
from flat_bug import logger as flatbug_logger, set_log_level # For flat-bug's internal logging

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883 # Standard unencrypted MQTT port
MQTT_TOPIC = "bsf_monitor/larvae_data" # <--- IMPORTANT: Make this topic unique for your project!

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
OUTPUT_DETECTION_DIR = "/home/pato/Documents/sdf/BSF-pi-script/detected_images"

# EasyOCR Settings
EASYOCR_LANGUAGES = ['en'] # Languages to load. 'en' for English.
EASYOCR_ALLOWLIST = '0123456789'
EASYOCR_BLOCKLIST = ''

# Script Timing
PROCESS_INTERVAL_SECONDS = 10 # How often to check for new images and process them

# --- Main Logic ---
def process_images_from_folder():
    """
    Looks for new images in the input directory, processes them, and publishes data to MQTT.
    """
    print(f"Checking for new images in {INPUT_IMAGE_DIR}...")
    images_found = False

    # Initialize EasyOCR reader once
    reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=False)

    # Corrected: Use DEFAULT_CFG directly, as it's already a dictionary
    # The read_cfg function expects a file path, not a dictionary.
    cfg = DEFAULT_CFG
    set_log_level(cfg.log_level)

    # Initialize Flat-Bug Predictor
    predictor = Predictor(cfg)
    
    # Ensure processed directory exists
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DETECTION_DIR, exist_ok=True)

    for filename in os.listdir(INPUT_IMAGE_DIR):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            images_found = True
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            print(f"Processing image: {image_path}")

            try:
                # Step 1: Read the image
                image_cv = cv2.imread(image_path)
                if image_cv is None:
                    print(f"Error: Could not read image {image_path}. Skipping.")
                    continue

                # Step 2: Perform OCR to get the tray number
                ocr_result = reader.readtext(image_path, allowlist=EASYOCR_ALLOWLIST)
                tray_number = None
                if ocr_result:
                    # Assuming the first detected number is the tray number
                    tray_number = ocr_result[0][1]
                    print(f"Detected Tray Number: {tray_number}")
                else:
                    print("No tray number detected.")
                    # Fallback or skip if no tray number can be read
                    continue

                # Step 3: Run YOLOv8 inference with the Flat-Bug predictor
                results = predictor.predict(image_path, single_scale=True)
                
                # Check if larvae were detected
                if results and results.get_count() > 0:
                    larvae_count = results.get_count()
                    avg_length = np.mean([l.length for l in results.get_larvae()]) if larvae_count > 0 else 0
                    avg_width = np.mean([l.width for l in results.get_larvae()]) if larvae_count > 0 else 0
                    avg_area = np.mean([l.area for l in results.get_larvae()]) if larvae_count > 0 else 0
                    avg_weight = np.mean([l.weight for l in results.get_larvae()]) if larvae_count > 0 else 0
                    
                    print(f"Detected {larvae_count} larvae for Tray {tray_number}.")
                    print(f"Average Length: {avg_length:.2f}mm")
                    print(f"Average Width: {avg_width:.2f}mm")
                    print(f"Average Area: {avg_area:.2f}mm²")
                    print(f"Average Weight: {avg_weight:.2f}mg")

                    # Step 4: Aggregate data into a JSON payload
                    payload = {
                        "tray_number": int(tray_number),
                        "larvae_count": larvae_count,
                        "avg_length_mm": round(avg_length, 2),
                        "avg_width_mm": round(avg_width, 2),
                        "avg_area_mm2": round(avg_area, 2),
                        "avg_weight_mg": round(avg_weight, 2),
                        "timestamp": datetime.now().isoformat()
                    }

                    # Step 5: Publish data to MQTT
                    try:
                        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
                        print(f"Data published successfully to MQTT broker.")
                    except Exception as mqtt_e:
                        print(f"Error publishing data to MQTT broker: {mqtt_e}")
                else:
                    print(f"No larvae detected for Tray {tray_number}. No data published to MQTT.")

            except Exception as e:
                print(f"Error during Flat-Bug inference or data aggregation for {image_path}: {e}")
                # It's good practice to log the full traceback for debugging
                traceback.print_exc()

            # Step 6: Move the processed image
            destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
            os.rename(image_path, destination_path)
            print(f"Moved processed image: {image_path} to {destination_path}")

    if not images_found:
        print("No new images found in the input folder.")

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- MQTT Setup ---
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()

    try:
        while True:
            process_images_from_folder()
            print(f"\nWaiting for {PROCESS_INTERVAL_SECONDS} seconds before checking again...")
            time.sleep(PROCESS_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nExiting program due to user interruption.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc() # Print traceback for unexpected errors
    finally:
        mqtt_client.loop_stop() # Stop the MQTT loop
        mqtt_client.disconnect() # Disconnect from the broker
        print("Program finished.")
