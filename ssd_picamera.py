# Import packages
from imutils.video.pivideostream import PiVideoStream

from keras import backend as K

from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast

import os
import cv2
import numpy as np
from threading import Thread
import sys
import io
import socketserver
from threading import Condition
from http import server
import logging
import datetime
import csv
import stat
from random import randrange
import argparse
import time

# Set up camera constants
#IM_WIDTH = 1280
#IM_HEIGHT = 720
#IM_WIDTH = 640    #Use smaller resolution for
#IM_HEIGHT = 480   #slightly faster framerate
IM_WIDTH = 300    #Use smaller resolution for
IM_HEIGHT = 300   #slightly faster framerate

PAGE="""\
<html>
<head>
<title>picamera MJPEG streaming demo</title>
</head>
<body>
<h1>PiCamera MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="""+str(IM_WIDTH)+""" height="""+str(IM_HEIGHT)+""" />
</body>
</html>
"""

class StreamingOutput(object):
    def __init__(self):
        self.frame = None
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            # New frame, copy the existing buffer's content and notify all
            # clients it's available
            self.buffer.truncate()
            with self.condition:
                self.frame = self.buffer.getvalue()
                self.condition.notify_all()
            self.buffer.seek(0)
        return self.buffer.write(buf)

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", type=int, default=0,
                help="whether or not to log debug messages")
ap.add_argument("-i", "--inference", type=int, default=1,
                help="do inference on video stream")
ap.add_argument("-c", "--captureclass", type=int, default=0,
                help="capture the individual inferenced images")
ap.add_argument("-l", "--logfile", type=str,help="log to file")
args = vars(ap.parse_args())

doinfer = args['inference']
dodebug = args['debug']
capture_class_images = args['captureclass']

# configure logging
FORMAT = '%(asctime)-15s %(levelname)-8s %(message)s'
loglevel = logging.INFO
if (dodebug):
    loglevel = logging.DEBUG
if (args["logfile"]):
    logging.basicConfig(filename='./ssd.log', filemode='w',level=loglevel, format=FORMAT)
else:
    logging.basicConfig(level=loglevel, format=FORMAT)

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
#from utils import label_map_util
#from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 90

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

ssd_img_height = 300
ssd_img_width = 300
normalize_coords = True
output_map = {0: 'background', 1: 'buddy', 2: 'jade', 3: 'lucy', 4: 'tim'}
# Set the threshold for detection
detection_threshold = 0.5
ssd_image_list = []
irand = randrange(0, 1000)

### Picamera ###
if camera_type == 'picamera':

    vs = PiVideoStream(resolution=(IM_WIDTH, IM_HEIGHT)).start()

    output = StreamingOutput()

    try:
        # prepare training csv
        if not os.path.isdir("/tmp/cats"):
            os.mkdir("/tmp/cats")
            os.mkdir("/tmp/cats/train")
            os.chmod("/tmp/cats", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
            os.chmod("/tmp/cats/train", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

        if not os.path.isfile("/tmp/cats/train/train2.csv"):
            with open('/tmp/cats/train/train2.csv', 'a') as outcsv:
                writer = csv.writer(outcsv)
                writer.writerow(['frame','xmin','xmax','ymin','ymax','class_id'])
                outcsv.close()

        today = datetime.datetime.now().strftime("%Y%m%d")

        ssd_counter = 0
        class_counter = 0
        address = ('', 10000)
        server = StreamingServer(address, StreamingHandler)

        logging.info("starting http server thread")
        thread = Thread(target=server.serve_forever)
        thread.daemon=True
        thread.start()
        logging.info("http server thread started")

        if (doinfer):
            logging.info("load model...")
            weights_path = './ssd7_cats_weights.h5'
            infer_model_path = './ssd7_cats_infer_fast.h5'

            # We need to create an SSDLoss object in order to pass that to the model loader.
            ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

            #K.clear_session() # Clear previous models from memory.

            model = load_model(infer_model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                                 'DecodeDetectionsFast': DecodeDetectionsFast,
                                                                 'compute_loss': ssd_loss.compute_loss})
            model.load_weights(weights_path, by_name=True)

            logging.info("model loaded")

        time.sleep(2.0)

        # loop over some frames...this time using the threaded stream
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()
            #frame = imutils.resize(frame, width=400)

            t1 = cv2.getTickCount()

            image_saved = False

            if (doinfer):
                # Compute the scale in order to draw bounding boxes on the full resolution
                # image.
                yscale = float(frame.shape[0]/ssd_img_height)
                xscale = float(frame.shape[1]/ssd_img_width)

                logging.debug("create cv2 array...")
                frame_resize = cv2.resize(frame, (ssd_img_width, ssd_img_height))
                image = img_to_array(frame_resize)
                image = np.expand_dims(image, axis=0)
                logging.debug("create cv2 array done")

                logging.debug("do prediction...")
                y_pred = model.predict(image)
                obj = [y_pred[k][y_pred[k,:,1] > detection_threshold] for k in range(y_pred.shape[0])]
                """
                obj = decode_detections(y_pred,
                                        confidence_thresh=0.5,
                                        iou_threshold=0.45,
                                        top_k=3,
                                        normalize_coords=normalize_coords,
                                        img_height=ssd_img_height,
                                        img_width=ssd_img_width)
                """

                #np.set_printoptions(precision=2, suppress=True, linewidth=90)
                logging.debug("do prediction done")

                logging.debug("grab images...")
                for infer in obj:
                    for box in infer:

                        # save the ssd image
                        if not image_saved:
                            frame_filename = "{}_{:03d}_{}_{:03d}".format(today, ssd_counter, 'cats', irand)
                            frame_path = "/tmp/cats/train/" + frame_filename + '.jpg'
                            cv2.imwrite(frame_path, frame_resize)
                            ssd_counter += 1
                            image_saved = True;

                        # Add bounding boxes to full resolution frame
                        xmin = int(xscale * box[-4])
                        ymin = int(yscale * box[-3])
                        xmax = int(xscale * box[-2])
                        ymax = int(yscale * box[-1])
                        id = int(box[0])
                        prob = float(box[1])

                        if xmin < 0:
                            xmin = 0
                        if ymin < 0:
                            ymin = 0
                        if xmax > IM_WIDTH:
                            xmax = IM_WIDTH-1
                        if ymax > IM_HEIGHT:
                            ymax = IM_HEIGHT-1

                        if capture_class_images:
                            dir = "/tmp/cats/train/" + frame_filename
                            if not os.path.isdir(dir):
                                os.mkdir(dir)
                                os.chmod(dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                            path = "/tmp/cats/train/{}/train_{}_{:03d}_{}_{:03d}_{:03d}_{:03d}_{:03d}_{:03d}.jpg".format(frame_filename, today, class_counter, output_map[id], irand, xmin, xmax, ymin, ymax)

                            crop = frame[ymin:ymax,xmin:xmax].copy()
                            cv2.imwrite(path, crop)
                            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                            class_counter += 1


                        stuff = "tag: {}:  prob:{:.2f}  xmin:{:03d}  ymin:{:03d}  xmax:{:03d}  ymax:{:03d}".format(output_map[id],prob,xmin,ymin,xmax,ymax)
                        logging.info(stuff)

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 2)
                        # Amount to offset the label/probability text above the bounding box.
                        text_offset = 15
                        # See https://docs.opencv.org/3.4.1/d6/d6e/group__imgproc__draw.html
                        # for more information about the cv2.putText method.
                        # Method signature: image, text, origin, font face, font scale, color,
                        # and tickness
                        cv2.putText(frame, "{}: {:.2f}%".format(output_map[id],prob * 100),
                                    (xmin, ymin-text_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 20), 2)

                        # create ssd entry
                        ssd_image_desc = [frame_filename+".jpg", xmin, xmax, ymin, ymax, id]
                        ssd_image_list.append(ssd_image_desc)

                logging.debug("grab images done")

            if (cv2.getTickCount()%100 == 0 or frame_rate_calc < 2.0):
                fr = "FPS: {0:.2f}".format(frame_rate_calc)
            cv2.putText(frame,fr,(30,50),font,1,(255,255,0),1)

            #imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            logging.debug("encoding buffer for browser...")
            r, buf = cv2.imencode(".jpg",frame)
            output.write(bytearray(buf))
            logging.debug("encoding buffer for browser done")

            # update the FPS counter
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1

            if image_saved:
                with open('/tmp/cats/train/train2.csv', 'a') as outcsv:
                    writer = csv.writer(outcsv)
                    writer.writerows(ssd_image_list)
                    ssd_image_list=[]
                    outcsv.close()

            # Press 'q' to quit
            """
            c = cv2.waitKey(1)
            if c == 'q':
                break
            elif c == 'w':
                vs.brightnessUp()
            elif c == 's':
                vs.brightnessDown()
            elif c == 'e':
                vs.contrastUp()
            elif c == 'd':
                vs.contrastDown()
            elif c == 'r':
                vs.staturationUp()
            elif c == 'f':
                vs.saturationDown()
            """
    finally:
        vs.stop()


