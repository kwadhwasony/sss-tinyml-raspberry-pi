#!/usr/bin/env python

#import device_patches       # Device specific patches for Jetson Nano (needs to be before importing cv2)

import cv2
import os
import sys, getopt
import signal
import time
from fix import ImageImpulseRunner
from picamera2 import Picamera2, Preview
import numpy as np

picam = {}
csi_cam_port = 'dev/video0'


runner = None
# if you don't want to see a camera preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False
    


def camera_init():
	global picam
	print('Opening CSI Camera with Picamera2 Library')
	picam = Picamera2()
	config = picam.create_preview_configuration()
	picam.configure(config)
	# todo: check if we need this
	#picam.start_preview(Preview.QTGL)
	picam.start()
	#print(len(frame_csi), len(frame_csi[0]))
	#picam.capture_file("test-python-csi.jpg")
	#rgba_image = PIL.Image.open(path_to_image)
	#rgb_image = frame_csi.convert('RGB')
	#return frame_csi

def now():
    return round(time.time() * 1000)

# method to convert rgba to rgb
# todo: check if we can actually use rgba directly with edge impulse sample code
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

def get_webcams():
    port_ids = []
    for port in range(5):
        print("Looking for a camera in port %s:" %port)
        camera = cv2.VideoCapture(port)
        if camera.isOpened():
            print("Camera was opened")
            time.sleep(1)
            print(camera.read())
            ret = camera.read()[0]
            if ret:
                backendName =camera.getBackendName()
                w = camera.get(3)
                h = camera.get(4)
                print("Camera %s (%s x %s) found in port %s " %(backendName,h,w, port))
                port_ids.append(port)
            else:
                print("Debug 0")
            camera.release()
        else: 
            print("Debug 1")
    return port_ids

def sigint_handler(sig, frame):
    print('Interrupted')
    if (runner):
        runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

def help():
    print('python classify.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

def main(argv):
    global picam
    camera_init()
    try:
        opts, args = getopt.getopt(argv, "h", ["--help"])
    except getopt.GetoptError:
        help()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            help()
            sys.exit()

    if len(args) == 0:
        help()
        sys.exit(2)

    model = args[0]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    modelfile = os.path.join(dir_path, model)

    print('MODEL: ' + modelfile)

    with ImageImpulseRunner(modelfile) as runner:
        try:
            print("Debug 0");
            model_info = runner.init()
            print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
            labels = model_info['model_parameters']['labels']
            #if len(args)>= 2:
                #videoCaptureDeviceId = int(args[1])
            #else:
                #port_ids = get_webcams()
                #if len(port_ids) == 0:
                    #raise Exception('Cannot find any webcams')
                #if len(args)<= 1 and len(port_ids)> 1:
                    #raise Exception("Multiple cameras found. Add the camera port ID as a second argument to use to this script")
                #videoCaptureDeviceId = int(port_ids[0])
            #camera = cv2.VideoCapture(videoCaptureDeviceId)
            #ret = camera.read()[0]
            #if ret:
                #backendName = camera.getBackendName()
                #w = camera.get(3)
                #h = camera.get(4)
                #print("Camera %s (%s x %s) in port %s selected." %(backendName,h,w, videoCaptureDeviceId))
                #camera.release()
            #else:
                #raise Exception("Couldn't initialize selected camera.")

            next_frame = 0 # limit to ~10 fps here
            
            
            
            #for res, img in runner.classifier(videoCaptureDeviceId):
            while True:
                
                img_csi = picam.capture_array()
                img_csi_rgb = rgba2rgb(img_csi)

                features, img = runner.get_features_from_image(img_csi_rgb)
                res = runner.classify(features)

                print("Debug 1");
                print(img)
                if (next_frame > now()):
                    time.sleep((next_frame - now()) / 1000)

                # print('classification runner response', res)

                if "classification" in res["result"].keys():
                    print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
                    for label in labels:
                        score = res['result']['classification'][label]
                        print('%s: %.2f\t' % (label, score), end='')
                    print('', flush=True)

                elif "bounding_boxes" in res["result"].keys():
                    print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
                    for bb in res["result"]["bounding_boxes"]:
                        print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
                        img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)

                if (show_camera):
                    cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) == ord('q'):
                        break

                next_frame = now() + 100
        finally:
            if (runner):
                runner.stop()

if __name__ == "__main__":
   main(sys.argv[1:])
