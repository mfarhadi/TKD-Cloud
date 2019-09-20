import threading
from frame_loader import Frame_reader
from TKD_detection import *
import torch.nn as nn
import torch.optim as optim

class F_loader(threading.Thread):
   def __init__(self, info):
      threading.Thread.__init__(self)
      self.info=info
   def run(self):

      print ("Starting " + str(self.info.threadID))

      Frame_reader(self.info)
      print("Exiting " + str(self.info.threadID))

class student():
    def __init__(self,threadID,TKD,oracle,source,opt,device):
        self.frame=None
        self.oracle=oracle
        self.TKD=TKD
        self.threadID = threadID
        self.source = source
        self.opt = opt
        self.device=device
        self.exitFlag=False
        self.loss=nn.MSELoss().cuda()
        self.optimizer = optim.Adam(TKD.parameters(), lr=0.01)


class student_detection(threading.Thread):

    def __init__(self, model, student):
        threading.Thread.__init__(self)
        self.model = model
        self.student=student
    def run(self):
        print("Starting student " + str(self.student.threadID))
        Fast_detection(self.model, self.student)
        print("Exiting student" + str(self.student.threadID))

class Oracle(threading.Thread):

    def __init__(self, model):
        threading.Thread.__init__(self)
        self.oracle=model
        self.frame=None
        self.feature=None
        self.info=None
    def run(self):

        Retraining(self.frame, self.feature, self.info,self.oracle)
