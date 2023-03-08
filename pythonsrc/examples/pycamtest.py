import time
from picamera2 import Picamera2, Preview
import cv2



import time

from picamera2 import Picamera2, Preview

picam2 = Picamera2()
picam2.start_preview(Preview.QTGL)

preview_config = picam2.create_preview_configuration()
capture_config = picam2.create_still_configuration()
picam2.configure(capture_config)

picam2.start()
time.sleep(2)

picam2.switch_mode_and_capture_file(capture_config, "test_full.jpg")

picam2.close()




#picam = Picamera2()

#config = picam.create_preview_configuration()
#picam.configure(config)
#print(config)
#picam.start_preview(Preview.QTGL)

#picam.start()
#array = picam.capture_array()
#print(array)
#time.sleep(2)
#picam.capture_file("test-python.jpg")
#picam.close()
