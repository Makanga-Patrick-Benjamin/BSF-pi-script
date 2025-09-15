# BSF-pi-script

A Bash script designed to simplify the installation and configuration of the BSF script on a Raspberry Pi or other Linux-based systems.

## Features
- Automates the setup process for BSF on a Raspberry Pi.
- Installs necessary dependencies.
- Configures the environment for optimal performance.
- Easy-to-use script with minimal user input required.

## Prerequisites
- A Raspberry Pi (or any Debian-based Linux system)
- `sudo` privileges
- Internet connection

## Installation
1. **Clone the repository**:
```bash
   git clone https://github.com/Makanga-Patrick-Benjamin/BSF-pi-script.git
   cd BSF-pi-script
```

2. **create a virtual environment**:
```bash
   python3 -m venv myenv #for python venv environment
   source myenv/bin/activate
```
```bash
   conda create --name myenv #for conda environment
   conda activate "myenv"
```
4. Install Dependencies:
   Navigate into the cloned directory. If your project uses Python dependencies, they should be listed in a requirements.txt file. Install them using pip:
```bash
   pip install -r requirements.txt
```
5.  Flat-Bug Model Configuration. Open and locate the "FLATBUG_MODEL_PATH" in image-read-flatbug.py and replace with the right model location on your raspberry pi:
- FLATBUG_MODEL_PATH = "/your_file_path_for_the_model_you_are_using/bestyolov8s.pt"

6.  Configuration. locate the file with the images you want to classify. you have to also provide as second location where ther are to move to as shown below(locate these lines of code as well):
- INPUT_IMAGE_DIR = "/home/pato/Documents/sdf/img" # <--- IMPORTANT: SET YOUR INPUT IMAGE FOLDER HERE!
- PROCESSED_IMAGE_DIR = "/home/pato/Documents/sdf/processed_images" # Directory to move processed images. Sort and change images here after processing.

7.  Execute the Script:
```bash
   python3 image-read-flatbug.py
```


## view content on webdash board
[https://soldierfly-fly-monitor.onrender.com](https://soldierfly-fly-monitor.onrender.com) *Dashboard layout*

**Default credentials:**
- **user name:** admin
- **password:** admin123