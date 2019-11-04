import queue
import threading
import time
import argparse
import time
from sys import platform
import torch

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
global exitFlag
exitFlag=[False]

from classes import *

import argparse
import os
import threading
import time
import torch
import torch.distributed as dist
import sys

class accuracy_live(threading.Thread):

    def __init__(self,opt):
        threading.Thread.__init__(self)
        self.info=opt
    def run(self):
        F1_score(self.info)

def F1_score(opt):

   img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

   device = torch_utils.select_device(force_cpu=ONNX_EXPORT)


   ################ Teacher ##########################

   o_weights, half = opt.o_weights, opt.half
   # Initialize model
   o_model = Darknet(opt.o_cfg, img_size)

   # Load weights
   if o_weights.endswith('.pt'):  # pytorch format
       o_model.load_state_dict(torch.load(o_weights, map_location=device)['model'])
   else:  # darknet format
       _ = load_darknet_weights(o_model, o_weights)

   # Eval mode
   o_model.to(device).eval()


   # Half precision
   half = half and device.type != 'cpu'  # half precision only supported on CUDA
   if half:
       o_model.half()

   print('listen for evaluation')
   s = socket.socket()
   s.bind((b'', 8000))

   s.listen(1)

   c, a = s.accept()
   loss_f=nn.MSELoss().cuda()
   while True:
       data = b''
       size = c.recv(4096)
       c.sendall(size)
       while len(data) < int(size):
           block = c.recv(4096)
           if not block: break
           data += block

       if sys.version_info.major < 3:
           img = pickle.loads(data)
       else:
           img = pickle.loads(data, encoding='bytes')

       c.sendall(size)

       data = b''
       size = c.recv(4096)
       c.sendall(size)

       while len(data) < int(size):
           block = c.recv(4096)
           if not block: break
           data += block
       if sys.version_info.major < 3:
           shallow_label = pickle.loads(data)
       else:
           shallow_label = pickle.loads(data, encoding='bytes')

       c.sendall(size)

       data = b''
       size = c.recv(4096)
       c.sendall(size)

       while len(data) < int(size):
           block = c.recv(4096)
           if not block: break
           data += block
       if sys.version_info.major < 3:
           loss = pickle.loads(data)
       else:
           loss = pickle.loads(data, encoding='bytes')

       c.sendall(size)


       frame = torch.from_numpy(img).cuda()
       pred, deep_T = o_model(frame)

       deep_label= non_max_suppression(pred, opt.conf_thres, opt.nms_thres)

       shallow_label=torch.from_numpy(shallow_label)
       correct = [0] * len(shallow_label)

       try:
           if deep_label[0] is not None:

               deep_label=deep_label[0].cpu()
               nl = deep_label.shape[0]
               detected = []
               tcls_tensor =deep_label[:,6]
               tcls = tcls_tensor.tolist() if nl else []
               tbox = deep_label[:, 0:4]


               for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(shallow_label):

                   # Break if all targets already located in image
                   if len(detected) == nl:
                       break

                   # Continue if predicted class not among image classes
                   if pcls.item() not in tcls:
                       continue

                   # Best iou, index between pred and targets

                   m = (pcls == tcls_tensor).nonzero().view(-1)
                   iou, bi = bbox_iou(pbox, tbox[m]).max(0)

                   # If iou > threshold and class is correct mark as correct
                   if iou > 0.4 and m[bi] not in detected:  # and pcls == tcls[bi]:
                       correct[i] = 1
                       detected.append(m[bi])
           tp=sum(x > 0 for x in correct)
           fp=len(correct)-tp
           recall = tp / (nl + 1e-16)
           precision = tp / (tp + fp)
           f1=2*(recall*precision)/(precision+recall)




           print('Recall:',recall,'Precision:',precision,'F1 score:',f1, 'Loss:', loss)
       except:
           print('ignore frame')








