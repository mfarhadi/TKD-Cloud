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

def Argos(opt):

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
           shallow_T1 = pickle.loads(data)
       else:
           shallow_T1 = pickle.loads(data, encoding='bytes')

       c.sendall(size)



       data = b''
       size = c.recv(4096)
       c.sendall(size)

       while len(data) < int(size):
           block = c.recv(4096)
           if not block: break
           data += block
       if sys.version_info.major < 3:
           shallow_T2 = pickle.loads(data)
       else:
           shallow_T2 = pickle.loads(data, encoding='bytes')

       c.sendall(size)

       shallow_T1 = torch.from_numpy(shallow_T1).cuda()
       shallow_T2 = torch.from_numpy(shallow_T2).cuda()


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
           loss=0


           loss += TKD_loss(shallow_T1, deep_T[0], loss_f)
           loss += TKD_loss(shallow_T2, deep_T[1], loss_f)

           print('Recall:',recall,'Precision:',precision,'F1 score:',f1, 'Loss:', float(loss.cpu()))
       except:
           print('ignore frame')
   # Notify threads it's time to exit
   input()
   for student_temp in students:
       student_temp.exitFlag=True
   for index in range(len(students)):
       del students[0]
   # Wait for all threads to complete
   for t in threads:
      t.join()
   print ("Exiting Main Thread")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s-cfg', type=str, default='cfg/yolov3-tiny.cfg', help='cfg file path')
    parser.add_argument('--o-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--s-weights', type=str, default='weights/yolov3-tiny.weights', help='path to weights file')
    parser.add_argument('--o-weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument("--backend", type=str, default='gloo',
                        help="Backend")
    parser.add_argument('-s', "--send", action='store_true',
                        help="Send tensor (if not specified, will receive tensor)")
    parser.add_argument("--master_addr", type=str,default='10.218.110.18',
                        help="IP address of master")
    parser.add_argument("--use_helper_threads", action='store_true',
                        help="Use multiple threads")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of current worker")
    parser.add_argument('-p', "--master_port", default=12344,
                        help="Port used to communicate tensors")
    parser.add_argument("--intra_server_broadcast", action='store_true',
                        help="Broadcast within a server")
    opt = parser.parse_args()


    with torch.no_grad():
        Argos(opt)







