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
from live_accuracy import *

def Argos(opt):

   img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

   device = torch_utils.select_device(force_cpu=ONNX_EXPORT)


   ################ STUDENT ##########################

   s_weights, half = opt.s_weights, opt.half

   # Initialize model
   s_model = Darknet(opt.s_cfg, img_size)

   s_model.feture_index=[6,8,12]
   # Load weights
   if s_weights.endswith('.pt'):  # pytorch format
       s_model.load_state_dict(torch.load(s_weights, map_location=device)['model'])
   else:  # darknet format
       _ = load_darknet_weights(s_model, s_weights)


   # Eval mode
   s_model.to(device).eval()

   # Half precision
   half = half and device.type != 'cpu'  # half precision only supported on CUDA
   if half:
       s_model.half()

   TKD_decoder = Darknet(opt.T_cfg, img_size)


   try:
       TKD_decoder.load_state_dict(torch.load(opt.T_weights, map_location=device)['model'])
   except:
       print('No TKD Decoder weights')
   if half:
       TKD_decoder.half()

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

   threadList = opt.source

   threads = []
   threadID = 1
   Remote_students=[]

   acc=accuracy_live(opt)
   acc.start()


   # Create new threads
   for tName in threadList:
      student_temp=Remote_student(threadID,TKD_decoder,s_model,o_model,opt,device)
      thread = Remote_student_update(student_temp)
      thread.start()
      Remote_students.append(thread)
      threadID += 1




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
    parser.add_argument('--s-cfg', type=str, default='cfg/yolov3-tiny.cfg', help='tiny cfg file path')
    parser.add_argument('--o-cfg', type=str, default='cfg/yolov3.cfg', help='Oracle cfg file path')
    parser.add_argument('--T-cfg', type=str, default='cfg/TKD_decoder_3out.cfg', help='TKD cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--s-weights', type=str, default='weights/yolov3-tiny.weights', help='tiny path to weights file')
    parser.add_argument('--o-weights', type=str, default='weights/yolov3.weights', help='Oracle path to weights file')
    parser.add_argument('--T-weights', type=str, default='weights/TKD.pt', help='TKD path to weights file')
    parser.add_argument('--source', type=str, default='0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', default=False, help='half precision FP16 inference')

    parser.add_argument("--backend", type=str, default='gloo',
                        help="Backend")
    parser.add_argument('-s', "--send", action='store_true',
                        help="Send tensor (if not specified, will receive tensor)")
    parser.add_argument("--master_addr", type=str,default='localhost',
                        help="IP address of master")
    parser.add_argument("--use_helper_threads", action='store_true',
                        help="Use multiple threads")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank of current worker")
    parser.add_argument('-p', "--master_port", default=12345,
                        help="Port used to communicate tensors")
    parser.add_argument("--intra_server_broadcast", action='store_true',
                        help="Broadcast within a server")

    opt = parser.parse_args()


    with torch.no_grad():
        Argos(opt)







