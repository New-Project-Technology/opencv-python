import picamera
import time
import datetime

with picamera.PiCamera() as camera:
	now = datetime.datetime.now()
	filename = now.strftime('%Y-%m%d %H:%M:%S')
	#camera.start_preview()
	#time.sleep(2)
	#camera.stop_preview()
	camera.capture(filename + '.jpg')
