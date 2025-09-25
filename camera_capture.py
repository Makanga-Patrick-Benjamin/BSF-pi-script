from time import sleep
from datetime import datetime
import subprocess
import os

def capture_image(DIR):
    if not os.path.exists(DIR):
        os.makedirs(DIR)
        print(f"Created directory: {DIR}")
        
    fileName= datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"
    filePath = os.path.join(DIR, fileName)
    cmd = "raspistill -t 1000 -o " + filePath

    subprocess.call(cmd, shell=True)   
    print('Image' + fileName)
    sleep(5)
