import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt # Import MQTT library
import json # To send data as JSON
import onnxruntime as ort

# --- Flat-Bug Model Imports (keep these lines for now, but we will not use Predictor) ---
# from flat_bug.predictor import Predictor
# from flat_bug.config import DEFAULT_CFG, read_cfg # For configuration if needed
# from flat_bug import logger as flatbug_logger, set_log_level # For flat-bug's internal logging

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
OUTPUT_DETECTION_DIR = "/home/pato/Documents/sdf/BSF-pi-script/detected_images" # Directory to save images with detections.
MODEL_PATH = "/home/pato/Documents/sdf/bestmodel.onnx" # Your Flat-Bug ONNX model path
CONFIDENCE_THRESHOLD = 0.25 # Confidence threshold for detections. Lower this if you are missing larvae.
PROCESS_INTERVAL_SECONDS = 5 # Time to wait before checking for new images again.

# --- Global objects ---
# Initialize MQTT client
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# Initialize EasyOCR reader
print("Initializing EasyOCR reader. This may download models on first run...")
# 'en' for English text, `allowlist='0123456789'` to recognize only integers
reader = easyocr.Reader(['en'], allowlist='0123456789')
print("EasyOCR reader initialized successfully for integer-only recognition.")

# Initialize ONNX Runtime session
print(f"Loading Flat-Bug model from: {MODEL_PATH} on device: cpu...")
try:
    # Create an inference session with the ONNX model
    ort_session = ort.InferenceSession(MODEL_PATH)
    print("Flat-Bug model loaded successfully.")

    # Get the input name and shape
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    print(f"Model input name: {input_name}, input shape: {input_shape}")

except Exception as e:
    print(f"Error loading Flat-Bug model: {e}")
    # Exit if the model fails to load
    exit()

def preprocess_image(image):
    """
    Prepares a single image for ONNX model inference.
    Resizes, normalizes, and reshapes the image to the model's expected format.
    """
    # Assuming the model expects 640x640 input. Adjust if your model is different.
    img_size = input_shape[2] # Typically 640
    
    # Resize the image to the expected input size
    resized_img = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # Convert the image from BGR (OpenCV default) to RGB
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to the range [0.0, 1.0] and transpose dimensions
    # The ONNX model expects the format (1, 3, height, width)
    normalized_img = rgb_img.astype(np.float32) / 255.0
    input_tensor = np.transpose(normalized_img, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor, resized_img # Return resized for drawing later

def postprocess_results(detections, resized_img, original_img_shape):
    """
    Processes the raw ONNX output to get bounding boxes, confidence scores,
    and class IDs. It then scales the boxes back to the original image size.
    """
    # The output format of the ONNX model can vary.
    # The common YOLOv8 ONNX output shape is (1, 84, 8400) where 84 is
    # [x, y, w, h, conf, class0, class1, ...]
    
    output = detections[0].squeeze() # Remove batch dimension

    # Find the confidence score and class ID for each detection
    conf_scores = output[4, :]
    class_ids = np.argmax(output[5:, :], axis=0)

    # Filter out low-confidence detections
    valid_detections = conf_scores > CONFIDENCE_THRESHOLD

    boxes = output[0:4, valid_detections].T
    conf_scores = conf_scores[valid_detections]
    class_ids = class_ids[valid_detections]

    # Convert boxes from normalized [x,y,w,h] to absolute [x1,y1,x2,y2]
    # This might need adjustment depending on your model's output format
    img_h, img_w = original_img_shape[:2]
    
    # YOLOv8 format: center_x, center_y, width, height
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2 # x1
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2 # y1
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]     # x2
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]     # y2

    # Scale boxes back to the original image dimensions
    scale_x = img_w / resized_img.shape[1]
    scale_y = img_h / resized_img.shape[0]
    
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    return boxes, conf_scores, class_ids

# --- Main Functions ---
def process_images_from_folder():
    """
    Main function to process images in a folder, perform OCR and detection,
    and publish data to MQTT.
    """
    print("Checking for new images to process...")
    images_found = False

    # Get a list of image files to process
    image_files = [f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Process images one by one
    for filename in image_files:
        images_found = True
        image_path = os.path.join(INPUT_IMAGE_DIR, filename)
        print(f"Processing new image: {image_path}")

        try:
            # Step 1: Read the image using OpenCV
            original_img = cv2.imread(image_path)
            if original_img is None:
                print(f"Warning: Could not read image at {image_path}. Skipping.")
                continue

            # Step 2: Extract tray number using EasyOCR
            tray_number = None
            try:
                # The OCR needs a grayscale image
                gray_image = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
                # Define a specific region of interest (ROI) for the tray number
                # Adjust these coordinates based on your image layout
                # Example: tray_roi = gray_image[y_start:y_end, x_start:x_end]
                # For this example, we'll try to find it on the whole image
                ocr_results = reader.readtext(gray_image)
                if ocr_results:
                    # Look for the first result that looks like a tray number (e.g., a single integer)
                    for (bbox, text, prob) in ocr_results:
                        if text.isdigit() and len(text) <= 3 and prob > 0.5: # Simple heuristic
                            tray_number = int(text)
                            print(f"Detected Tray Number: {tray_number} with confidence {prob:.2f}")
                            break
            except Exception as e:
                print(f"Error during OCR: {e}")
            
            if tray_number is None:
                print("Could not reliably detect tray number. Skipping this image.")
                continue

            # Step 3: Flat-Bug inference with ONNX Runtime
            print(f"Starting larvae detection for Tray {tray_number}...")
            
            # Pre-process the image for the model
            input_tensor, resized_img_for_onnx = preprocess_image(original_img)
            
            # Run inference
            detections = ort_session.run(None, {input_name: input_tensor})

            # Post-process the results
            bboxes, scores, class_ids = postprocess_results(detections, resized_img_for_onnx, original_img.shape)

            # Assuming class_id 0 is 'larvae'.
            larvae_count = sum(1 for c in class_ids if c == 0)
            print(f"Detected {larvae_count} larvae in Tray {tray_number}.")

            # Step 4: Aggregate and publish data to MQTT
            payload = {
                "tray_number": tray_number,
                "timestamp": datetime.now().isoformat(),
                "larvae_count": larvae_count,
            }

            if larvae_count > 0:
                # Draw detections on the image for visual verification
                detected_img = original_img.copy()
                for bbox, score in zip(bboxes, scores):
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(detected_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(detected_img, f'{score:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the detected image
                output_filename = f"tray_{tray_number}_larvae_detected.jpg"
                output_path = os.path.join(OUTPUT_DETECTION_DIR, output_filename)
                os.makedirs(OUTPUT_DETECTION_DIR, exist_ok=True)
                cv2.imwrite(output_path, detected_img)
                print(f"Saved image with detections to: {output_path}")

                try:
                    mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
                    print(f"Data published successfully to MQTT broker.")
                except Exception as mqtt_e:
                    print(f"Error publishing data to MQTT broker: {mqtt_e}")
            else:
                print(f"No data to publish for Tray {tray_number} (no larvae detected).")

            # Move the processed image to the 'processed' folder
            destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
            os.rename(image_path, destination_path)
            print(f"Moved processed image: {image_path} to {destination_path}")

        except Exception as e:
            print(f"Error during processing for {image_path}: {e}")
            import traceback
            traceback.print_exc()

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