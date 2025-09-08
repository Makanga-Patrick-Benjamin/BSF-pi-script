import time
import cv2
import os
import easyocr
from datetime import datetime
import numpy as np
import paho.mqtt.client as mqtt
import json
import onnxruntime as ort

# --- ONNX Runtime Model Configuration ---
# You need to have your ONNX model file and a labels file for class names.
ONNX_MODEL_PATH = "/home/pato/Documents/sdf/bestmodel.onnx"
# LABELS_PATH = "path/to/your/labels.txt" # Uncomment and provide if you have a labels file.

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
INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img"
PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/processed_images"
OUTPUT_DETECTION_DIR = "/home/pato/Documents/sdf/BSF-pi-script/detected_larvae" # Directory to save images with detections drawn on them.
PROCESS_INTERVAL_SECONDS = 5 # How often to check for new images.

# --- Initialize MQTT Client ---
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.on_connect = on_connect
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# --- Initialize ONNX Runtime Session and Model ---
try:
    # Create an InferenceSession with the ONNX model.
    # The `providers` argument can be used to specify the hardware (e.g., 'CPUExecutionProvider').
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    
    # Get the input name of the model.
    input_name = session.get_inputs()[0].name
    
    print(f"ONNX model loaded successfully from {ONNX_MODEL_PATH}")

    # You may also want to get the model's expected input shape and output names here.
    input_shape = session.get_inputs()[0].shape
    output_names = [output.name for output in session.get_outputs()]
    print(f"Model input shape: {input_shape}")
    print(f"Model output names: {output_names}")

    # Optional: Load labels if you have a labels file.
    # with open(LABELS_PATH, 'r') as f:
    #     labels = [line.strip() for line in f.readlines()]
    # print(f"Labels loaded: {labels}")

except Exception as e:
    print(f"Error initializing ONNX Runtime session: {e}")
    exit()

def preprocess_image(image):
    """
    Preprocesses the image for the ONNX model.
    This function should be customized based on your specific model's requirements.
    A common format is resizing to a fixed size, transposing to C,H,W format,
    and normalizing pixel values.
    """
    # Resize the image to the model's expected input size
    # This example assumes the model expects a 224x224 image.
    model_input_size = (input_shape[3], input_shape[2]) # (width, height)
    resized_image = cv2.resize(image, model_input_size)
    
    # Change HxWxC to CxHxW format (transpose the channels).
    resized_image = np.transpose(resized_image, (2, 0, 1))

    # Add a batch dimension (e.g., from (3, 224, 224) to (1, 3, 224, 224)).
    input_tensor = resized_image[np.newaxis, :, :, :]

    # Normalize the image (e.g., by dividing by 255.0).
    input_tensor = input_tensor.astype(np.float32) / 255.0

    return input_tensor

def process_images_from_folder():
    """
    Checks for new images in the input folder and processes them.
    """
    images_found = False
    if not os.path.exists(PROCESSED_IMAGE_DIR):
        os.makedirs(PROCESSED_IMAGE_DIR)

    image_files = sorted([f for f in os.listdir(INPUT_IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if image_files:
        images_found = True
        print(f"Found {len(image_files)} new images to process.")

        for filename in image_files:
            image_path = os.path.join(INPUT_IMAGE_DIR, filename)
            print(f"Processing image: {image_path}")

            try:
                # Read the image using OpenCV
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Could not read image file {image_path}. Skipping.")
                    continue

                # Preprocess the image for the model
                input_tensor = preprocess_image(image)

                # Run inference with ONNX Runtime
                # Pass the preprocessed image and get the raw output.
                raw_outputs = session.run(None, {input_name: input_tensor})

                # Post-process the output
                # This part is highly dependent on your specific model.
                # Here we assume the model output is a single tensor of probabilities.
                predictions = raw_outputs[0]
                predicted_class = np.argmax(predictions)
                confidence = np.max(predictions)

                print(f"Predicted class index: {predicted_class}, Confidence: {confidence:.2f}")

                # Simulate a detection result for the rest of the code logic.
                if confidence > 0.5: # Use a threshold.
                    detected_larvae = {
                        "larvae_count": int(predicted_class),  # Example: use the class index as the count.
                        "confidence": float(confidence),
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"Larvae detection result: {detected_larvae}")

                    # --- OCR Part (unchanged from original logic) ---
                    # The OCR part is kept as per the user's request to maintain logic.
                    # It reads the tray number from a specified region of interest (ROI).
                    tray_roi = image[0:100, 0:300]
                    reader = easyocr.Reader(['en'])
                    results = reader.readtext(tray_roi)
                    
                    tray_number = None
                    for (bbox, text, prob) in results:
                        if text.strip().isdigit() and prob > 0.5:
                            tray_number = int(text.strip())
                            print(f"Detected Tray Number: {tray_number}")
                            break

                    # --- Data Aggregation and MQTT Publishing (unchanged from original logic) ---
                    payload = {
                        "timestamp": datetime.now().isoformat(),
                        "tray_id": tray_number,
                        "image_path": image_path,
                        "larvae_count": detected_larvae.get("larvae_count", 0),
                        "detection_confidence": detected_larvae.get("confidence", 0.0)
                    }

                    try:
                        mqtt_client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
                        print(f"Data published successfully to MQTT broker.")
                    except Exception as mqtt_e:
                        print(f"Error publishing data to MQTT broker: {mqtt_e}")
                else:
                    print(f"No data to publish for this image (low confidence).")

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