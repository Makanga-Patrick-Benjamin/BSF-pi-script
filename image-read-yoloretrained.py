import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
from ultralytics import YOLO # Import YOLO
import paho.mqtt.client as mqtt # Import MQTT library
import json # To send data as JSON

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883 # Standard unencrypted MQTT port
MQTT_TOPIC = "bsf_monitor/larvae_data" # <--- IMPORTANT: Make this topic unique for your project!
                                      # E.g., "your_username/bsf_monitor/larvae_data"

# --- Callbacks for MQTT Client ---
def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

# --- Configuration ---
INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img" # <--- IMPORTANT: SET YOUR INPUT IMAGE FOLDER HERE!
PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/BSF-pi-script/ocr_processed_images" # Directory to move processed images

# EasyOCR Settings
EASYOCR_LANGUAGES = ['en'] # Languages to load. 'en' for English.
EASYOCR_ALLOWLIST = '0123456789'
EASYOCR_BLOCKLIST = ''

# Script Timing
PROCESS_INTERVAL_SECONDS = 10 # How often to check for new images and process them

# YOLOv8 Model Configuration
YOLOV8_MODEL_PATH = "/home/pato/Documents/sdf/YoloRetrain.pt" # <--- IMPORTANT: SET PATH TO YOUR TRAINED YOLOv8 MODEL

# Calibration Factor (pixels per millimeter)
PIXELS_PER_MM = 20.0 # Placeholder: Adjust this value based on your calibration!

# --- Initialize EasyOCR Reader ---
print("Initializing EasyOCR reader. This may download models on first run...")
try:
    reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=False)
    print("EasyOCR reader initialized successfully for integer-only recognition.")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    print("Please ensure you have an internet connection for the first run to download models.")
    exit()

# --- Initialize YOLOv8 Model ---
print(f"Loading YOLOv8 model from: {YOLOV8_MODEL_PATH}...")
try:
    model = YOLO(YOLOV8_MODEL_PATH)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    print("Please ensure your model path is correct and Ultralytics is installed.")
    exit()

# --- Initialize MQTT Client ---
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) # Specify API version
# mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start() # Start the loop in a background thread to handle re-connections
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")
    exit()

# --- Image Preprocessing for EasyOCR ---
def preprocess_image_for_easyocr(image):
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    return denoised

# --- Text Extraction Function (EasyOCR) ---
def extract_text_with_easyocr(image_path):
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
                        print(f"  Recognized integer '{integer_value}' with confidence: {float(confidence):.2f}")
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
    x1, y1, x2, y2 = bbox
    length_px = abs(y2 - y1)
    width_px = abs(x2 - x1)

    if mask is not None:
        area_px = np.sum(mask)
    else:
        area_px = length_px * width_px

    length_mm = length_px / PIXELS_PER_MM
    width_mm = width_px / PIXELS_PER_MM
    area_sq_mm = area_px / (PIXELS_PER_MM ** 2)

    WEIGHT_PER_SQ_MM = 6.67 # Placeholder: Adjust this based on your empirical data!
    estimated_weight_mg = area_sq_mm * WEIGHT_PER_SQ_MM

    return length_mm, width_mm, area_sq_mm, estimated_weight_mg

# --- Main Processing Loop ---
def process_images_from_folder():
    os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)

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
                print(f"Moved image (no tray number): {image_path} to {destination_path}")
                continue

            print(f"Running YOLOv8 inference on {image_path}...")
            larvae_data_to_send = []
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

                            larvae_data_to_send.append({
                                "tray_number": tray_number,
                                "length": round(length_mm, 2),
                                "width": round(width_mm, 2),
                                "area": round(area_sq_mm, 2),
                                "weight": round(estimated_weight_mg, 2),
                                "count": 1 # Each entry is for one larva
                            })
                            print(f"  Larva {larva_id + 1}: L={length_mm:.2f}mm, W={width_mm:.2f}mm, A={area_sq_mm:.2f}mm², Wt={estimated_weight_mg:.2f}mg (Conf: {larva_confidence:.2f}%)")
                    else:
                        print(f"No larvae detected by YOLOv8 in Tray {tray_number}.")

                # Send aggregated data via MQTT
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

                    print(f"Publishing aggregated data for Tray {tray_number} to MQTT topic '{MQTT_TOPIC}': {payload}")
                    try:
                        # Convert payload to JSON string
                        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1) # QoS 1 for at least once delivery
                        print(f"Data published successfully to MQTT broker.")
                    except Exception as mqtt_e:
                        print(f"Error publishing data to MQTT broker: {mqtt_e}")
                else:
                    print(f"No larvae detected for Tray {tray_number}. No data published to MQTT.")

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
        mqtt_client.loop_stop() # Stop the MQTT loop
        mqtt_client.disconnect() # Disconnect from the broker
        print("Program finished.")