import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt
import json
import onnxruntime as ort

# --- MQTT Configuration ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "bsf_monitor/larvae_data"

# --- Callbacks for MQTT Client ---
def on_connect(client, userdata, flags, rc, properties):
    """Callback function for when the MQTT client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}\n")

# --- Configuration ---
# Set directories to be relative to the script's location
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img" # <--- IMPORTANT: SET YOUR INPUT IMAGE FOLDER HERE!
PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/processed_images" # Directory to move processed images. Sort and change images here after processing.
OUTPUT_DETECTION_DIR = "/home/pato/Documents/sdf/BSF-pi-script/detected_images" # Images with bounding boxes
MODEL_PATH = "/home/pato/Documents/sdf/bestmodel.onnx"

PROCESS_INTERVAL_SECONDS = 5  # Check for new images every 5 seconds

# --- Class for ONNX Inference ---
class ONNXPredictor:
    """
    Handles ONNX model loading and inference.
    NOTE: The pre-processing and post-processing logic MUST be customized
    to match the specific ONNX model's input and output requirements.
    """
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def run_inference(self, image):
        """
        Processes the image and runs inference.
        You MUST fill in this part with your model's specific logic.

        Args:
            image (np.ndarray): The input image (e.g., loaded with cv2.imread).

        Returns:
            A tuple of (boxes, confidence_scores, masks).
            The format of these depends entirely on your model's output.
        """
        # Placeholder for pre-processing logic
        # Example for a YOLO-like model (you will need to adjust this)
        input_shape = self.session.get_inputs()[0].shape[2:]
        input_tensor = cv2.resize(image, tuple(input_shape))
        input_tensor = input_tensor.transpose((2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = input_tensor.astype('float32') / 255.0

        # Run inference
        raw_outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Placeholder for post-processing logic
        # You must implement the code to parse raw_outputs and convert them
        # into bounding boxes, confidence scores, and masks.
        # This will be very specific to your model.
        # Example:
        # boxes, confs, masks = self.post_process(raw_outputs, image.shape)
        # For now, we return empty lists to allow the script to run without crashing.
        return [], [], []

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

# --- Helper Functions ---
def calculate_area(mask):
    return np.sum(mask > 0)

def calculate_approximate_weight(area_pixels, reference_weight=0.04):
    """
    Calculates the approximate weight of a larva based on its area in pixels.
    This is an arbitrary function based on the original script's logic.
    """
    # Adjust this based on your camera calibration and object properties.
    PIXELS_PER_MM = 10  # Example value.
    area_mm2 = area_pixels / (PIXELS_PER_MM ** 2)
    return area_mm2 * reference_weight

def process_images_from_folder():
    """
    Main function to process images from the input folder.
    """
    print(f"Scanning for new images in: {INPUT_IMAGE_DIR}")
    images_found = False

    # Check if directories exist and create them if they don't
    create_directory_if_not_exists(INPUT_IMAGE_DIR)
    create_directory_if_not_exists(PROCESSED_IMAGE_DIR)
    create_directory_if_not_exists(OUTPUT_DETECTION_DIR)

    # Load ONNX model once
    try:
        predictor = ONNXPredictor(MODEL_PATH)
        print("ONNX model loaded successfully.")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Initialize EasyOCR reader
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        print("EasyOCR reader initialized.")
    except Exception as e:
        print(f"Error initializing EasyOCR reader: {e}")
        return

    for filename in os.listdir(INPUT_IMAGE_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            images_found = True
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            print(f"Processing image: {image_path}")
            
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_path}. Skipping.")
                continue

            # Read the tray number from the image using EasyOCR
            tray_number = "UNKNOWN"
            try:
                # Assuming the tray number is in a fixed region
                roi_image = image[100:200, 100:400]
                result_easyocr = reader.readtext(roi_image, detail=0)
                if result_easyocr:
                    tray_number = " ".join(result_easyocr)
                    print(f"Detected Tray Number: {tray_number}")
            except Exception as e:
                print(f"EasyOCR Error: {e}")

            # --- Larvae detection and analysis with ONNX ---
            larvae_data = []
            annotated_image = image.copy()
            
            try:
                # You must customize the run_inference method in ONNXPredictor
                # to get meaningful boxes, confs, and masks.
                boxes, confs, masks = predictor.run_inference(image)
                
                if boxes:
                    # Logic from the original script to iterate through results
                    for i in range(len(boxes)):
                        box = boxes[i]
                        conf = confs[i]
                        mask = masks[i]

                        # Calculate properties from the mask
                        larva_area = calculate_area(mask)
                        larva_weight = calculate_approximate_weight(larva_area)
                        
                        # Find bounding box for the mask
                        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
                        
                        # Append data to the list
                        larvae_data.append({
                            "id": i,
                            "bbox": [int(x), int(y), int(w), int(h)],
                            "confidence": float(conf),
                            "area_pixels": float(larva_area),
                            "weight_g": float(larva_weight)
                        })

                        # Draw bounding box and confidence on the image
                        cv2.rectangle(annotated_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(annotated_image, f"{conf:.2f}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                        # Create and apply the color mask
                        color_mask = np.zeros_like(image, dtype=np.uint8)
                        color_mask[mask > 0] = (255, 0, 0)
                        annotated_image = cv2.addWeighted(annotated_image, 1, color_mask, 0.5, 0)

                    print(f"Detected {len(larvae_data)} larvae.")
                    
                    # Save the annotated image
                    annotated_filename = f"detected_{os.path.basename(filename)}"
                    annotated_path = os.path.join(OUTPUT_DETECTION_DIR, annotated_filename)
                    cv2.imwrite(annotated_path, annotated_image)
                    print(f"Saved detected image to {annotated_path}")

                    # Publish data to MQTT
                    payload = {
                        "tray_id": tray_number,
                        "timestamp": datetime.now().isoformat(),
                        "total_larvae_count": len(larvae_data),
                        "larvae_details": larvae_data
                    }
                    try:
                        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
                        print("Data published successfully to MQTT broker.")
                    except Exception as mqtt_e:
                        print(f"Error publishing data to MQTT broker: {mqtt_e}")
                else:
                    print(f"No data to publish for Tray {tray_number} (no larvae detected).")
            
            except Exception as e:
                print(f"Error during ONNX inference or data aggregation for {image_path}: {e}")
                import traceback
                traceback.print_exc()

            destination_path = os.path.join(PROCESSED_IMAGE_DIR, filename)
            os.rename(image_path, destination_path)
            print(f"Moved processed image: {image_path} to {destination_path}")

    if not images_found:
        print("No new images found in the input folder.")

# --- Main Execution Block ---
if __name__ == "__main__":
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()

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