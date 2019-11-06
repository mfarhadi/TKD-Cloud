import cv2
import numpy
import threading

class motion_info:
    def __init__(self):
        self.static_back=None
        self.motion_status=True
        self.gray=None

class motion_detection(threading.Thread):

    def __init__(self,frame,info):
        threading.Thread.__init__(self)
        self.frame=numpy.copy(frame)
        self.info=info
    def run(self):

        # Initializing motion = 0(no motion)
        motion = 0

        # Converting color image to gray_scale image
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        # Converting gray scale image to GaussianBlur
        # so that change can be find easily
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        self.info.gray = gray
        # In first iteration we assign the value
        # of static_back to our first frame
        if self.info.static_back is None:
            self.info.static_back = gray
        else:
            # Difference between static background
            # and current frame(which is GaussianBlur)
            diff_frame = cv2.absdiff(self.info.static_back, gray)

            # If change in between static background and

            # current frame is greater than 30 it will show white color(255)
            thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
            thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
            self.info.gray = gray
            if thresh_frame.sum()>0:
                self.info.motion_status= True
            else:
                self.info.motion_status = False