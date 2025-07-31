import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt # Import MQTT library
import json # To send data as JSON

# --- Flat-Bug Model Imports ---
# You need to have the flat-bug library installed: pip install flat-bug
# And the model weights (e.g., flat_bug_M.pt) downloaded.
from flat_bug.predictor import Predictor
from flat_bug.config import DEFAULT_CFG, read_cfg # For configuration if needed
from flat_bug import logger as flatbug_logger, set_log_level # For flat-bug's internal logging

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883 # Standard unencrypted MQTT port
MQTT_TOPIC = "bsf_monitor/larvae_data" # <--- IMPORTANT: Make this topic unique for your project!
                                      # E.g., "your_username/bsf_monitor/larvae_data"

# --- Callbacks for MQTT Client ---
# FIX: Updated on_connect signature for paho.mqtt.CallbackAPIVersion.VERSION2
def on_connect(client, userdata, flags, rc, properties):
    """Callback function for when the MQTT client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

# --- Configuration ---
INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img" # <--- IMPORTANT: SET YOUR INPUT IMAGE FOLDER HERE!
PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/BSF-pi-script/ocr_processed_images" # Directory to move processed images

# EasyOCR Settings
EASYOCR_LANGUAGES = ['en'] # Languages to load. 'en' for English.
EASYOCR_ALLOWLIST = '0123456789' # Only allow digits for tray number recognition
EASYOCR_BLOCKLIST = ''

# Script Timing
PROCESS_INTERVAL_SECONDS = 10 # How often to check for new images and process them

# Flat-Bug Model Configuration
# <--- IMPORTANT: SET PATH TO YOUR DOWNLOADED FLAT-BUG MODEL WEIGHTS (.pt file)
FLATBUG_MODEL_PATH = "/home/pato/Documents/sdf/best1024.pt" # Updated based on your error log
# Choose device: 'cuda:0' for GPU (if available), 'cpu' for CPU
FLATBUG_DEVICE = "cpu" # Recommended for Raspberry Pi or systems without dedicated GPU
FLATBUG_DTYPE = "float32" # Use float32 for CPU, float16 for GPU if supported

# Calibration Factor (pixels per millimeter)
# Placeholder: Adjust this value based on your camera setup and calibration!
# You'll need to determine how many pixels correspond to 1 millimeter in your images.
PIXELS_PER_MM = 20.0

# --- Initialize EasyOCR Reader ---
print("Initializing EasyOCR reader. This may download models on first run...")
try:
    # Set gpu=False for EasyOCR if running on CPU-only device like most Raspberry Pis
    reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=False)
    print("EasyOCR reader initialized successfully for integer-only recognition.")
except Exception as e:
    print(f"Error initializing EasyOCR: {e}")
    print("Please ensure you have an internet connection for the first run to download models.")
    exit()

# --- Initialize Flat-Bug Model ---
print(f"Loading Flat-Bug model from: {FLATBUG_MODEL_PATH} on device: {FLATBUG_DEVICE}...")
try:
    # Load default configuration or a custom one if needed
    flatbug_config = DEFAULT_CFG
    # You can set flatbug_logger level to DEBUG for more verbose output from flat-bug
    # set_log_level("DEBUG")

    # Initialize the Predictor
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
# Using CallbackAPIVersion.VERSION2 for future compatibility and to suppress deprecation warning
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start() # Start the loop in a background thread to handle re-connections
except Exception as e:
    print(f"Failed to connect to MQTT broker: {e}")
    exit()

# --- Image Preprocessing for EasyOCR ---
def preprocess_image_for_easyocr(image):
    """Converts image to grayscale and applies median blur for OCR."""
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 3) # Helps remove noise that might interfere with OCR
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
        # Use allowlist to restrict OCR to only digits
        results = reader.readtext(processed_image, allowlist=EASYOCR_ALLOWLIST)

        for (bbox, text, confidence) in results:
            if text.strip(): # Check if text is not empty
                # Filter out non-digit characters from the recognized text
                cleaned_text = ''.join(filter(str.isdigit, text.strip()))
                if cleaned_text:
                    try:
                        integer_value = int(cleaned_text) # Convert to int to validate
                        extracted_integers.append(str(integer_value)) # Store as string
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
        # For simplicity, take the first recognized integer.
        # More robust logic might involve choosing based on confidence or position.
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
        # Sum of pixels in the mask where the larva is present
        area_px = np.sum(mask)
    else:
        # Fallback to bounding box area if no mask is provided
        area_px = length_px * width_px

    # Convert pixel measurements to real-world units using calibration factor
    length_mm = length_px / PIXELS_PER_MM
    width_mm = width_px / PIXELS_PER_MM
    area_sq_mm = area_px / (PIXELS_PER_MM ** 2)

    # Placeholder for weight estimation: Adjust this based on your empirical data!
    # This is a simple linear relationship; a more complex model might be needed.
    WEIGHT_PER_SQ_MM = 6.67 # Example: mg per square millimeter of larva area
    estimated_weight_mg = area_sq_mm * WEIGHT_PER_SQ_MM

    return length_mm, width_mm, area_sq_mm, estimated_weight_mg

# --- Main Processing Loop ---
def process_images_from_folder():
    """
    Monitors the input directory for new images, processes them,
    and publishes aggregated data to MQTT.
    """
    os.makedirs(INPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(PROCESSED_IMAGE_DIR, exist_ok=True)

    print(f"\n--- Checking for new images in {INPUT_IMAGE_DIR} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    images_found = False
    for filename in os.listdir(INPUT_IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            images_found = True
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            print(f"Processing image: {image_path}")

            # Step 1: Extract Tray Number using EasyOCR
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

            # If no valid tray number, just move the image and continue
            if tray_number is None:
                destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
                os.rename(image_path, destination_path)
                print(f"Moved image (no tray number detected): {image_path} to {destination_path}")
                continue

            # Step 2: Run Flat-Bug Inference for Larva Detection and Measurement
            print(f"Running Flat-Bug inference on {image_path}...")
            larvae_data_to_send = []
            total_count = 0

            try:
                # Perform inference using the flat-bug predictor
                # pyramid_predictions is good for robust detection but can be slow.
                # For faster inference, you might use a simpler prediction method if available
                # or adjust scale_before/single_scale parameters.
                prediction_results = flatbug_predictor.pyramid_predictions(
                    image_path,
                    scale_increment=2/3, # Default for pyramid_predictions
                    scale_before=1.0,    # No initial downscaling before detection
                    single_scale=False   # Use multi-scale prediction
                )

                # --- FIX: Accessing results from TensorPredictions object ---
                # Check if any larvae were detected by checking if the 'boxes' tensor is not empty
                # Also ensure 'boxes' attribute exists before trying to access its length or content.
                if prediction_results and hasattr(prediction_results, 'boxes') and prediction_results.boxes is not None and len(prediction_results.boxes) > 0:
                    total_count = len(prediction_results.boxes)
                    print(f"Found {total_count} larvae in Tray {tray_number}.")

                    # Iterate through each detected instance using indices
                    for larva_id in range(total_count):
                        # Extract bounding box (xyxy format)
                        bbox_xyxy = prediction_results.boxes[larva_id].tolist() # Convert tensor to list
                        
                        # Access confidence scores from the 'confs' attribute of the prediction_results object
                        larva_confidence = prediction_results.confs[larva_id].item() # Convert tensor scalar to float

                        mask = None
                        # Check if masks are present and if there's a mask for the current larva_id
                        # FIX: Access the 'data' attribute of the Masks object for the raw mask tensor
                        if hasattr(prediction_results, 'masks') and prediction_results.masks is not None and len(prediction_results.masks) > larva_id:
                            # Access the specific Masks object for this larva
                            larva_mask_object = prediction_results.masks[larva_id]
                            # Access the raw data tensor from the Masks object, then convert to numpy and cast
                            mask = larva_mask_object.data.cpu().numpy().astype(np.uint8)
                            # The mask is typically already at the original image resolution or scaled appropriately
                            # for area calculation. No explicit resizing here unless flat-bug's output dictates it.

                        # Calculate metrics using the extracted bbox and mask
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
                    print(f"No larvae detected by Flat-Bug in Tray {tray_number}.")

                # Step 3: Aggregate and Send Data via MQTT
                if total_count > 0:
                    # Calculate averages for the detected larvae in this image/tray
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
                        "count": total_count # Total count for this tray from this image
                    }

                    print(f"Publishing aggregated data for Tray {tray_number} to MQTT topic '{MQTT_TOPIC}': {payload}")
                    try:
                        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1) # QoS 1 for at least once delivery
                        print(f"Data published successfully to MQTT broker.")
                    except Exception as mqtt_e:
                        print(f"Error publishing data to MQTT broker: {mqtt_e}")
                else:
                    print(f"No larvae detected for Tray {tray_number}. No data published to MQTT.")

            except Exception as e:
                print(f"Error during Flat-Bug inference or data aggregation for {image_path}: {e}")
                # It's good practice to log the full traceback for debugging
                import traceback
                traceback.print_exc()

            # Step 4: Move the processed image
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
        traceback.print_exc() # Print traceback for unexpected errors
    finally:
        mqtt_client.loop_stop() # Stop the MQTT loop
        mqtt_client.disconnect() # Disconnect from the broker
        print("Program finished.")
