import os
import time
from picamera2 import Picamera2, Preview
import libcamera

def capture_image(output_path, file_prefix="BSFimg"):
    """
    Captures a high-quality image using a Raspberry Pi camera module,
    focuses the camera, and saves it to a specified location with a timestamp.

    Args:
        output_path (str): The directory where the image will be saved.
        file_prefix (str): The prefix for the image filename.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Created directory: {output_path}")

    # Initialize the camera
    try:
        picam2 = Picamera2()
        
        # Configure the camera for a high-resolution still image
        picam2.configure(picam2.create_still_configuration(
            main={"size": (1600, 1200)},  # Adjust resolution as needed
            lores={"size": (640, 480)},  # Lower resolution for preview
            display="lores"
        ))

        print("Camera initialized.")
        
        # Start the camera and perform autofocus
        picam2.start()
        print("Camera started. Performing autofocus...")
        picam2.set_controls({"AfMode": libcamera.controls.AfModeEnum.Auto, "AfTrigger": libcamera.controls.AfTriggerEnum.Start})
        
        # Wait for autofocus to complete
        time.sleep(2) 
        
        # Create a unique filename with a timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{file_prefix}_{timestamp}.jpg"
        full_path = os.path.join(output_path, filename)

        # Capture the image
        print(f"Capturing image and saving to {full_path}...")
        picam2.capture_file(full_path)
        print("Image captured successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop the camera and close the instance
        if 'picam2' in locals() and picam2.is_open:
            picam2.stop()
            print("Camera stopped.")

if __name__ == '__main__':
    # This is an example of how to call the function
    # You can change the directory to your desired location
    image_directory = "/home/pato/Pictures/BSF_images"
    capture_image(image_directory)