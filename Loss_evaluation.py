
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
import argparse
import time
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from torch.autograd import Variable

import torch.optim as optim
from loss_preparation import TKD_loss
import torch.distributed as dist
import os
import scipy.io as sio
import numpy as np


import threading

global exitFlag
exitFlag=[False]

import os


from classes import *


def Argos(opt):

   img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

   device = torch_utils.select_device(force_cpu=ONNX_EXPORT)

   data = opt.data
   data_dict = parse_data_cfg(data)

   ################ STUDENT ##########################

   s_weights, half = opt.s_weights, opt.half

   # Initialize model
   s_model = Darknet(opt.s_cfg, img_size)

   s_model.feture_index=[8,12]
   # Load weights
   if s_weights.endswith('.pt'):  # pytorch format
       s_model.load_state_dict(torch.load(s_weights, map_location=device)['model'])
   else:  # darknet format
       _ = load_darknet_weights(s_model, s_weights)


   # Eval mode
   s_model.to(device).eval()
   model=s_model
   # Half precision
   half = half and device.type != 'cpu'  # half precision only supported on CUDA
   if half:
       s_model.half()

   TKD_decoder = Darknet('cfg/TKD_decoder.cfg', img_size)


   #if s_weights.endswith('.pt'):  # pytorch format
   TKD_decoder.load_state_dict(torch.load('weights/TKD.pt', map_location=device)['model'])

   hyp = {'giou': 1.582,  # giou loss gain
          'cls': 27.76,  # cls loss gain  (CE=~1.0, uCE=~20)
          'cls_pw': 1.446,  # cls BCELoss positive_weight
          'obj': 21.35,  # obj loss gain (*=80 for uBCE with 80 classes)
          'obj_pw': 3.941,  # obj BCELoss positive_weight
          'iou_t': 0.2635,  # iou training threshold
          'lr0': 0.002324,  # initial learning rate (SGD=1E-3, Adam=9E-5)
          'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
          'momentum': 0.97,  # SGD momentum
          'weight_decay': 0.0004569,  # optimizer weight decay
          'hsv_s': 0.5703,  # image HSV-Saturation augmentation (fraction)
          'hsv_v': 0.3174,  # image HSV-Value augmentation (fraction)
          'degrees': 1.113,  # image rotation (+/- deg)
          'translate': 0.06797,  # image translation (+/- fraction)
          'scale': 0.1059,  # image scale (+/- gain)
          'shear': 0.5768}  # image shear (+/- deg)

   TKD_decoder.hyp = hyp  # attach hyperparameters to model
   TKD_decoder.nc=int(data_dict['classes'])
   TKD_decoder.arc = opt.arc

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

   ################## Oracle for inference ###################

   Oracle_model = Darknet(opt.o_cfg, img_size)

   # Load weights
   if o_weights.endswith('.pt'):  # pytorch format
       Oracle_model.load_state_dict(torch.load(o_weights, map_location=device)['model'])
   else:  # darknet format
       _ = load_darknet_weights(Oracle_model, o_weights)

   # Eval mode
   Oracle_model.to(device).eval()

   # Half precision
   half = half and device.type != 'cpu'  # half precision only supported on CUDA
   if half:
       Oracle_model.half()

   threadList = opt.source

   threads = []
   threadID = 1
   students=[]



   info=student(threadID,TKD_decoder,o_model,opt.source,opt,dist,device)

   # Configure run

   nc = 9  # number of classes

   seen = 0
   model.eval()
   coco91class = coco80_to_coco91_class()
   s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
   p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.

   jdict, stats, ap, ap_class = [], [], [], []


   iou_thres = 0.5


   for source in info.source:

       webcam = source == '0' or source.startswith('rtsp') or source.startswith('http')
       streams = source == 'streams.txt'

       model.eval()

       info.TKD.eval().cuda()

       # Set Dataloader

       if streams:
           torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
           dataset = LoadStreams(source, img_size=info.opt.img_size, half=info.opt.half)
       elif webcam:
           stream_img = True
           dataset = LoadWebcam(source, img_size=info.opt.img_size, half=info.opt.half)
       else:
           save_img = True
           dataset = LoadImages(source, img_size=info.opt.img_size, half=info.opt.half)


       # Run inference
       info.frame = torch.zeros([1, 3, info.opt.img_size, info.opt.img_size])
       oracle_T = Oracle()
       info.oracle.train().cuda()


       counter=0
       confidence=0.001
       records=np.zeros((1000,2))
       for path, img, im0s, vid_cap in dataset:

           info.collecting = True
           # Get detections


           counter+=1
           info.frame[0, :, 0:img.shape[1], :] = torch.from_numpy(img)
           info.frame = info.frame.cuda()
           pred, _, feature = model(info.frame)
           info.TKD.img_size = info.frame.shape[-2:]
           pred_TKD, p = info.TKD(feature)

           Oracle_model.train()
           T_out = Oracle_model(info.frame)

           t1=time.time()
           richOutput = [Variable(T_out[0].data, requires_grad=False), Variable(T_out[1].data, requires_grad=False)]
           loss=0

           for i in range(2):
               loss += TKD_loss(p[i], richOutput[i], info.loss)
           t2=time.time()
           info.TKD.train()
           pred= info.TKD(feature)
           Oracle_model.eval()
           labels,_=Oracle_model(info.frame)
           t3=time.time()
           labels = non_max_suppression(labels, confidence, 0.5)

           labels=labels[0]

           if labels is not None:
               labels = labels[:, [4, 6, 0, 1, 2, 3]].round()
               labels[:, 2:] = xyxy2xywh(labels[:, 2:])
               labels[:, 2:] = labels[:, 2:] / 416
               labels[:, 0] = labels[:, 0] * 0
           if labels is not None:
               loss, loss_items = compute_loss(pred, labels, info.TKD)
               t4=time.time()
               print(labels.shape[0],t2-t1,t4-t3)
               records[labels.shape[0],:]=[t2-t1,t4-t3]
           if counter%100==0:
               if confidence<0.2:
                   confidence*=2
               elif confidence<0.9:
                   confidence+=0.1
           if labels.shape[0]==1:
               break
           info.TKD.eval()


       file = open('loss_time'+'.txt', 'a')
       for i in range(500):
           if records[i,0]!=0:
               file.write('\n'+str(i)+','+str(records[i,0]*1000)+','+str(records[i,1]*1000))
       file.close()





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s-cfg', type=str, default='cfg/yolov3-tiny.cfg', help='cfg file path')
    parser.add_argument('--o-cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--s-weights', type=str, default='weights/yolov3-tiny.weights', help='path to weights file')
    parser.add_argument('--o-weights', type=str, default='weights/yolov3.weights', help='path to weights file')
    parser.add_argument('--source', type=str, default=['/media/common/DATAPART1/datasets/UCF_Crimes/Videos/Training_Normal_Videos_Anomaly/Normal_Videos425_x264.mp4'], help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--arc', type=str, default='defaultpw', help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument("--backend", type=str, default='gloo',
                        help="Backend")
    parser.add_argument('-s', "--send", action='store_true',
                        help="Send tensor (if not specified, will receive tensor)")
    parser.add_argument("--master_addr", type=str,default='10.218.110.18',
                        help="IP address of master")
    parser.add_argument("--use_helper_threads", action='store_true',
                        help="Use multiple threads")
    parser.add_argument("--rank", type=int, default=1,
                        help="Rank of current worker")
    parser.add_argument('-p', "--master_port", default=12345,
                        help="Port used to communicate tensors")
    parser.add_argument("--intra_server_broadcast", action='store_true',
                        help="Broadcast within a server")

    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        Argos(opt)