import time
from picamera2 import Picamera2, Preview
import cv2

picam = Picamera2()

config = picam.create_preview_configuration()
picam.configure(config)

picam.start_preview(Preview.QTGL)

picam.start()
array = picam.capture_array()
print(array)
time.sleep(2)
picam.capture_file("test-python.jpg")
picam.close()
