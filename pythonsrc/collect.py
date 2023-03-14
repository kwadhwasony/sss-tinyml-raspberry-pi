# imports
import cv2
import numpy as np
import time
import os
import sys, getopt
import signal
import asyncio
from azure.iot.device import Message
from azure.iot.device.aio import IoTHubDeviceClient
from edge_impulse_linux.image import ImageImpulseRunner
from picamera2 import Picamera2, Preview
import numpy as np

# globals
picam = {}
runner = None
azure_client = None
# connection string pointing towards tanmoys iot hub device
#CONNECTION_STRING = "HostName=sonydemoiothub.azure-devices.net;DeviceId=sonytestdevice;SharedAccessKey=p/c31bhBeVelkeYLcx7Bws5BY5tBYXym3v1Tg7xTK/U="
# connection string pointing towards iot hub device on kunal.wadhwa@sony.com account
CONNECTION_STRING = "HostName=testhubname0.azure-devices.net;DeviceId=RPiDevice0;SharedAccessKey=hv82yeuIj9GTTXnAhuw8HWk7NGUOp9NU4NO/0NOMb9E="
PAYLOAD = '{{"X": {x_start}, "Y": {y_start}, "x":{height}, "y":{width}}}'

# if you don't want to see a camera preview, set this to False
show_camera = True
if (sys.platform == 'linux' and not os.environ.get('DISPLAY')):
    show_camera = False

# method to convert rgba to rgb
# note1: this method only does matrix transforms and does not indeed check
#	if the input is indeed a valid rgb format value
# note2: only works for unit8 type values of RGB.
def rgba2rgb(rgba, background=(255,255,255)):
    row, col, ch = rgba.shape

	# if already only 3 channel, means it is rgb (or similar format), return as is
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

# get time 
def now():
    return round(time.time() * 1000)

# interrupt handler
# todo: check if we need to destroy camera class here
def sigint_handler(sig, frame):
    print('Interrupted')
    global picam
    if (runner):
        runner.stop()
    # todo: check if pycam is open before we close it
    picam.close()
    sys.exit(0)
    
# exit signal initalized
signal.signal(signal.SIGINT, sigint_handler)

# method to init camera 
def camera_init():
	global picam
	print('Opening CSI Camera with Picamera2 Library')
	picam = Picamera2()
	# picam.close()
	#preview_config = picam2.create_preview_configuration()
	#capture_config = picam2.create_still_configuration()
	#picam2.configure(capture_config)
	#config = picam.create_preview_configuration()
	#config = picam.create_still_configuration()
	config = picam.create_still_configuration(main={"size": (1920, 1080)})
	#config = picam.create_still_configuration(main={"size": (1080, 720)})
	#camera_config = picam2.create_still_configuration(main={"size": (1920, 1080)}, lores={"size": (640, 480)}, display="lores")
	picam.configure(config)
	picam.start()
	# todo: add a way to check if initialized properly

# method to get image from camera
#	pi camera v1.3 return rgba frame
def capture_camera():
	frame_csi = picam.capture_array()
	return frame_csi
	
# method to get rgb frame from pi camera v1.3 
def capture_camera_rgb():
	frame_rgba = capture_camera()
	frame_rgb = rgba2rgb(frame_rgba)
	return frame_rgb

# todo: implement properly
# right now only shows how to run the python code, nothing else is display
def help():
    print('python demo.py <path_to_model.eim> <Camera port ID, only required when more than 1 camera is present>')

# method to initialize azure client
def azure_client_init():
	global azure_client
	azure_client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
	# todo: should return success failure

async def azure_client_send_payload(payload):
	global azure_client
	print('sending payload to azure iot hub:', payload)
	await azure_client.send_message(payload)

# main method
async def main(argv):
	start = 110
	iterator = start
	global picam
	global azure_client

	# check for valid arguments
	try:
		opts, args = getopt.getopt(argv, "h", ["--help"])
	except getopt.GetoptError:
		help()
		sys.exit(2)

	for opt, arg in opts:
		if opt in ('-h', '--help'):
			help()
			sys.exit()

	# if not arguments were given, exit since we do not know here to model file is
	if len(args) == 0:
		help()
		sys.exit(2)

	# initialze edge-impulse-related things (get model file into code and init)
	model = args[0]
	dir_path = os.path.dirname(os.path.realpath(__file__))
	modelfile = os.path.join(dir_path, model)
	print('model file: ' + modelfile)
	
	# inialize camera
	camera_init()

	# initialize azure iot related things
	azure_client_init()
	
	with ImageImpulseRunner(modelfile) as runner:
		try:
			model_info = runner.init()
			print('Loaded runner for "' + model_info['project']['owner'] + ' / ' + model_info['project']['name'] + '"')
			labels = model_info['model_parameters']['labels']

			# todo: change this to make sure we are able to run at optimum fps
			next_frame = 0 # limit to ~10 fps here

			# todo: change this loop to work only when camera is available
			while True:
				# get image array from the camera (rgba)
				img_csi = picam.capture_array()
				print(len(img_csi), len(img_csi[0]))
				# change that to rgb format
				# img_csi_rgb = rgba2rgb(img_csi)
				img_csi_rgb = img_csi
				print(len(img_csi_rgb), len(img_csi_rgb[0]))
				# feed to edge impulse methods
				features, img = runner.get_features_from_image(img_csi_rgb)
				res = runner.classify(features)
				print(len(img), len(img[0]))

				# tradition delay is induced here
				# todo: remove/optimize this for best frame rate
				#if (next_frame > now()):
				#	time.sleep((next_frame - now()) / 1000)
				
				# if there is classification in keys()
				# todo: check if we need all these printed results of if we need this if elif block
				#if "classification" in res["result"].keys():
				#	print('Result (%d ms.) ' % (res['timing']['dsp'] + res['timing']['classification']), end='')
				#	for label in labels:
				#		score = res['result']['classification'][label]
				#		print('%s: %.2f\t' % (label, score), end='')
				#	print('', flush=True)
				#elif "bounding_boxes" in res["result"].keys():
				#	print('Found %d bounding boxes (%d ms.)' % (len(res["result"]["bounding_boxes"]), res['timing']['dsp'] + res['timing']['classification']))
				#	for bb in res["result"]["bounding_boxes"]:
				#		print('\t%s (%.2f): x=%d y=%d w=%d h=%d' % (bb['label'], bb['value'], bb['x'], bb['y'], bb['width'], bb['height']))
				#		img = cv2.rectangle(img, (bb['x'], bb['y']), (bb['x'] + bb['width'], bb['y'] + bb['height']), (255, 0, 0), 1)
						
				#if "bounding_boxes" in res["result"].keys():
				#	# get the box with the max value, save as top_dog
				#	top_dog = res["result"]["bounding_boxes"][0]
				#	payload = PAYLOAD.format(x_start=top_dog["x"], y_start=top_dog["y"], height=top_dog["height"], width=top_dog["width"])
				#	await azure_client_send_payload(payload)
						
				# todo: remove this entirely for the demo. We do not even need to spend time to check the if condition every time
				# this is only for visualization at the edge
				iterator = iterator + 1
				if (show_camera):
					#cv2.imshow('edgeimpulse', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
					# dst = cv2.cvtColor(img, code)
					cv2.imwrite('images/test' + str(iterator) + '.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
					if (iterator == start + 10):
						break

				#next_frame = now() + 100

		finally:
			if (runner):
				runner.stop()	
	
if __name__ == "__main__":
	asyncio.run(main(sys.argv[1:]))
