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
    def __init__(self,threadID,TKD,oracle,source,opt,dist,device):
        self.frame=None
        self.oracle=oracle
        self.TKD=TKD
        self.threadID = threadID
        self.source = source
        self.opt = opt
        self.device=device
        self.exitFlag=False
        self.loss=nn.MSELoss().cuda()
        self.dist=dist
        self.optimizer = optim.Adam(TKD.parameters(), lr=0.001)


class Remote_student():
    def __init__(self,threadID,TKD,tinymodel,oracle,opt,device):
        self.frame=None
        self.oracle=oracle
        self.model=tinymodel
        self.TKD=TKD
        self.threadID = threadID
        self.opt = opt
        self.device=device
        self.exitFlag=False
        self.loss=nn.MSELoss().cuda()
        self.optimizer = optim.Adam(TKD.parameters(), lr=0.001)


class student_detection(threading.Thread):

    def __init__(self, model, student):
        threading.Thread.__init__(self)
        self.model = model
        self.student=student
    def run(self):
        print("Starting student " + str(self.student.threadID))
        Fast_detection(self.model, self.student)
        print("Exiting student" + str(self.student.threadID))

class Remote_student_update(threading.Thread):

    def __init__(self, student):
        threading.Thread.__init__(self)
        self.student=student
    def run(self):
        print("Starting student " + str(self.student.threadID))
        server_Retraining(self.student)
        print("Exiting student" + str(self.student.threadID))


