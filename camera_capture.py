import time
from picamera2 import Picamera2, Preview
from libcamera import controls

def capture_image(file_location, image_number):
    """
    Captures a clear image using the Raspberry Pi camera and saves it.

    Args:
        file_location (str): The directory to save the image.
        image_number (int): A unique number for the image filename.
    """
    try:
        # Initialize the camera
        cam = Picamera2()
        
        # Configure the camera with a high-resolution preview and a capture configuration
        # You can adjust these values based on your camera model and desired resolution.
        preview_config = cam.create_preview_configuration()
        cam.configure(preview_config)

        # Start the camera
        cam.start()
        
        # Wait for the camera to warm up and auto-focus
        # You can adjust the auto-focus mode if needed
        cam.set_controls({"AfMode": controls.AfModeEnum.Auto, "AfTrigger": controls.AfTriggerEnum.Start})
        time.sleep(2)  # Give it time to focus and stabilize

        # Generate a unique filename
        filename = f"BSFimg{image_number}.jpg"
        full_path = f"{file_location}/{filename}"
        
        # Capture the image
        cam.capture_file(full_path)
        print(f"Image saved to: {full_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop and close the camera gracefully
        if 'cam' in locals():
            cam.stop()
            cam.close()

if __name__ == "__main__":
    # This block only runs when the script is executed directly.
    # When called from another script, the 'capture_image' function is used.
    
    # Define your save location and a unique number
    save_directory = "/home/pato/Pictures"  # Change this to your desired location
    image_counter = int(time.time())  # This should be managed by the calling script

    # Take a picture
    capture_image(save_directory, image_counter)
    
    # If you want to take another picture, increment the counter
    # and call the function again
    # image_counter += 1
    # capture_image(save_directory, image_counter)