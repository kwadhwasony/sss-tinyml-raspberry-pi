import cv2
import time
from picamera2 import Picamera2, Preview
import PIL.Image
import numpy as np

usb_cam_port = '/dev/video0'
csi_cam_port = 'dev/video2'

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )


print('Opening USB Cam with CV2 Library')
print('Trying Camera Port: ' + usb_cam_port)
cap = cv2.VideoCapture(usb_cam_port, cv2.CAP_V4L)
ret, frame = cap.read()
print(ret, frame)
print(len(frame), len(frame[0]))
cv2.imwrite("test-python-usb.jpg", frame)

print('Opening CSI Camera with Picamera2 Library')
picam = Picamera2()

config = picam.create_preview_configuration()
picam.configure(config)

picam.start_preview(Preview.QTGL)

picam.start()
frame_csi = picam.capture_array()
print(frame_csi)
print(len(frame_csi), len(frame_csi[0]))
picam.capture_file("test-python-csi.jpg")

#rgba_image = PIL.Image.open(path_to_image)
#rgb_image = frame_csi.convert('RGB')

frame_csi_2 = rgba2rgb(frame_csi)
cv2.imwrite("test-python-csi-2.jpg", frame_csi_2)

picam.close()
