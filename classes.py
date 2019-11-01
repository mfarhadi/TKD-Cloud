import threading
from frame_loader import Frame_reader
from TKD_detection import *
import torch.nn as nn
import torch.optim as optim
import socket
import numpy
import time

try:
    import cPickle as pickle
except ImportError:
    import pickle

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
        self.network=False
        self.precision=False
        self.socket=None


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

class Remote_precision(threading.Thread):

    def __init__(self, image,detection, info):
        threading.Thread.__init__(self)
        self.image=image
        self.result=detection.cpu().numpy()
        self.info=info
    def run(self):
        print("ya ali")
        sock = socket.socket()
        sock.connect((self.info.opt.master_addr, 8000))
        serialized_data = pickle.dumps(self.image, protocol=2)
        sock.sendall(serialized_data)
        sock.close()
        sock.connect((self.info.opt.master_addr, 8000))
        serialized_data = pickle.dumps(self.result, protocol=2)
        sock.sendall(serialized_data)
        sock.close()




