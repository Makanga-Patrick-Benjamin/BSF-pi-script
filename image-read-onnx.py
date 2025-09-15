import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt
import json
import onnxruntime as ort
import glob
from pathlib import Path

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
PROCESSED_IMG_DIR = "/home/pato/Documents/sdf/processed_images" # Directory to move processed images. Sort and change images here after processing.
OUTPUT_DETECTION_DIR = "/home/pato/Documents/sdf/BSF-pi-script/detected_images" # Images with bounding boxes
MODEL_PATH = "/home/pato/Documents/sdf/bestmodel.onnx"

PROCESS_INTERVAL_SECONDS = 5  # Check for new images every 5 seconds

CONF_THRESHOLD = 0.5  # Confidence threshold for object detection
EASYOCR_LANGUAGES = ['en']
EASYOCR_RECOGNIZE_CONF = 0.5

class ONNXPredictor:
    """Handles inference with an ONNX model."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self._load_model()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _load_model(self):
        """Loads the ONNX model and sets up the inference session."""
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=['CPUExecutionProvider'])
            print("ONNX model loaded successfully.")
            print("Using CPU. Note: This module is much faster with a GPU.")
        except Exception as e:
            print(f"Error loading ONNX model: {e}")
            raise

    def run_inference(self, image_path):
        """
        Runs inference on a single image and returns detected objects.
        
        Args:
            image_path (Path): Path to the image file.

        Returns:
            list: A list of detected objects (larvae), each a dictionary with keys
                  'bounding_box' and 'confidence'.
        """
        try:
            # Load and preprocess the image
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")

            # --- MODEL PRE-PROCESSING: THIS IS A PLACEHOLDER ---
            # You must adapt this section to match your model's specific requirements.
            # Common steps include resizing, normalizing, and changing channel order.
            # The example below assumes a common YOLO-like model input.
            
            # Example for a YOLOv8-like model:
            # 1. Resize image to a fixed size (e.g., 640x640)
            input_shape = self.session.get_inputs()[0].shape
            input_width, input_height = input_shape[3], input_shape[2]
            
            img_resized = cv2.resize(image, (input_width, input_height))

            # 2. Normalize and transpose for ONNX model input
            input_data = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            input_data = input_data.transpose(2, 0, 1)  # Change to C, H, W
            input_data = input_data.astype('float32') / 255.0  # Normalize to [0, 1]
            input_data = input_data[None, :, :, :]  # Add batch dimension

            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            
            # --- MODEL POST-PROCESSING: THIS IS A PLACEHOLDER ---
            # You must adapt this section to match your model's specific output format.
            # The raw output from the model needs to be parsed into meaningful data.
            
            detected_larvae = []
            
            # Example for a YOLOv8-like model with a single output tensor:
            # The raw output needs to be transposed and filtered.
            predictions = outputs[0].transpose()
            
            # Filtering based on confidence and non-maximum suppression (NMS)
            for prediction in predictions:
                scores = prediction[4:]
                class_id = scores.argmax()
                confidence = scores[class_id]

                if confidence > CONF_THRESHOLD:
                    # Bounding box coordinates
                    x_center, y_center, width, height = prediction[:4]
                    
                    # Convert to original image coordinates
                    original_height, original_width = image.shape[:2]
                    x_factor = original_width / input_width
                    y_factor = original_height / input_height

                    xmin = int((x_center - width / 2) * x_factor)
                    ymin = int((y_center - height / 2) * y_factor)
                    xmax = int((x_center + width / 2) * x_factor)
                    ymax = int((y_center + height / 2) * y_factor)

                    detected_larvae.append({
                        "class_id": int(class_id),
                        "confidence": float(confidence),
                        "bounding_box": [xmin, ymin, xmax, ymax]
                    })
            
            # Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
            # This is a critical step, but its implementation depends on the library used.
            # You may need to use a library like `torchvision.ops.nms` or a custom function.
            
            return detected_larvae

        except Exception as e:
            print(f"Error during ONNX inference for {image_path}: {e}")
            return []

class EasyOCRReader:
    """Handles text recognition with EasyOCR."""
    def __init__(self, languages):
        try:
            self.reader = easyocr.Reader(languages)
            print("EasyOCR reader initialized.")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            raise

    def read_tray_number(self, image_path):
        """
        Reads the tray number from an image.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            str: The detected tray number or "UNKNOWN" if not found.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_path}")

            # Crop the top-right corner where the tray number is located
            h, w, _ = image.shape
            crop_img = image[0:int(h * 0.2), int(w * 0.7):w]

            # Perform OCR on the cropped image
            results = self.reader.readtext(crop_img)

            for (bbox, text, prob) in results:
                # Filter for results with high confidence that contain only digits
                if prob > EASYOCR_RECOGNIZE_CONF and text.isdigit():
                    return text.strip()
            
            return "UNKNOWN"

        except Exception as e:
            print(f"Error reading tray number from {image_path}: {e}")
            return "UNKNOWN"

class MQTTPublisher:
    """Manages MQTT connection and publishing."""
    def __init__(self, broker, port, topic):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect

    def on_connect(self, client, userdata, flags, rc):
        """Callback for when the client connects to the broker."""
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}\n")

    def on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects."""
        if rc != 0:
            print(f"Unexpected disconnection. Reconnecting...")

    def connect_and_loop(self):
        """Starts the MQTT client loop."""
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"Error connecting to MQTT Broker: {e}")
            raise

    def publish_data(self, data):
        """Publishes data to the specified MQTT topic."""
        try:
            self.client.publish(self.topic, json.dumps(data), qos=1)
            print("Data published successfully.")
        except Exception as e:
            print(f"Error publishing data: {e}")

    def disconnect(self):
        """Disconnects the MQTT client."""
        self.client.loop_stop()
        self.client.disconnect()

def process_images():
    """Main function to process images in a directory."""
    try:
        # Ensure processed images directory exists
        PROCESSED_IMG_DIR.mkdir(exist_ok=True)
        
        # Initialize components
        onnx_predictor = ONNXPredictor(MODEL_PATH)
        ocr_reader = EasyOCRReader(EASYOCR_LANGUAGES)
        mqtt_publisher = MQTTPublisher(MQTT_BROKER, MQTT_PORT, TOPIC)
        mqtt_publisher.connect_and_loop()

        while True:
            print(f"Scanning for new images in: {IMG_DIR}")
            image_files = glob.glob(f"{IMG_DIR}/*.jpg")
            
            if not image_files:
                print("No new images found. Waiting...")
                time.sleep(10)
                continue

            for image_path_str in image_files:
                image_path = Path(image_path_str)
                print(f"Processing image: {image_path}")

                try:
                    # 1. Analyze for larvae
                    larvae_detections = onnx_predictor.run_inference(image_path)
                    
                    # 2. Extract tray number
                    tray_number = ocr_reader.read_tray_number(image_path)

                    larvae_count = len(larvae_detections)
                    
                    if larvae_count > 0:
                        payload = {
                            "tray_number": tray_number,
                            "larvae_count": larvae_count,
                            "detections": [
                                {
                                    "bounding_box": det["bounding_box"],
                                    "confidence": det["confidence"]
                                } for det in larvae_detections
                            ],
                            "timestamp": time.time()
                        }
                        mqtt_publisher.publish_data(payload)
                        print(f"Data published for Tray {tray_number} ({larvae_count} larvae detected).")
                    else:
                        print(f"No data to publish for Tray {tray_number} (no larvae detected).")
                    
                    # Move the processed image
                    new_path = PROCESSED_IMG_DIR / image_path.name
                    image_path.rename(new_path)
                    print(f"Moved processed image: {image_path} to {new_path}")
                
                except Exception as e:
                    print(f"Failed to process image {image_path}. Error: {e}")
            
            time.sleep(5)  # Wait before checking for new images again

    except KeyboardInterrupt:
        print("Process interrupted by user. Shutting down.")
    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
    finally:
        if 'mqtt_publisher' in locals():
            mqtt_publisher.disconnect()
            print("Disconnected from MQTT Broker.")
        print("Script terminated.")

if __name__ == "__main__":
    process_images()