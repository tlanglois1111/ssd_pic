# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread

class PiVideoStream:
    # Set up camera constants
    #IM_WIDTH = 1280
    #IM_HEIGHT = 720
    IM_WIDTH = 640    #Use smaller resolution for
    IM_HEIGHT = 480   #slightly faster framerate

    def __init__(self, resolution=(IM_WIDTH, IM_HEIGHT), framerate=20):
        # initialize the camera and stream
        self.camera = PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size=resolution)
        self.rawCapture.truncate(0)

        self.stream = self.camera.capture_continuous(self.rawCapture,format="rgb", use_video_port=False)

        # initialize the frame and the variable used to indicate
        # if the thread should be stopped
        self.frame = None
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        for f in self.stream:
            # grab the frame from the stream and clear the stream in
            # preparation for the next frame
            self.frame = f.array
            self.rawCapture.truncate(0)

            # if the thread indicator variable is set, stop the thread
            # and resource camera resources
            if self.stopped:
                self.stream.close()
                self.rawCapture.close()
                self.camera.close()
                return

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def brightnessUp(self):
        self.camera.contrast = min(self.camera.brightness + 1, 100)
    def brightnessDown(self):
        self.camera.contrast = min(self.camera.brightness - 1, 0)

    def contrastUp(self):
        self.camera.contrast = min(self.camera.contrast + 1, 100)
    def contrastDown(self):
        self.camera.contrast = min(self.camera.contrast - 1, -100)

    def saturationUp(self):
        self.camera.contrast = min(self.camera.saturation + 1, 100)
    def saturationDown(self):
        self.camera.contrast = min(self.camera.saturation - 1, -100)
