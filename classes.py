import threading
from frame_loader import Frame_reader
from argos_detection import *

class F_loader(threading.Thread):
   def __init__(self, info):
      threading.Thread.__init__(self)
      self.info=info
   def run(self):

      print ("Starting " + str(self.info.threadID))

      Frame_reader(self.info)
      print("Exiting " + str(self.info.threadID))

class student():
    def __init__(self,threadID,source,opt,device):
        self.frame=None
        self.collecting=True
        self.ready=False
        self.results=None
        self.model=None
        self.threadID = threadID
        self.source = source
        self.opt = opt
        self.device=device
        self.exitFlag=False


class student_detection(threading.Thread):

    def __init__(self, model, students, threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.model = model
        self.students=students
    def run(self):
        print("Starting student " + str(self.threadID))
        Fast_detection(self.model, self.students)
        print("Exiting student" + str(self.threadID))