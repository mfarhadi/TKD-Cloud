
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



import threading

global exitFlag
exitFlag=[False]

import os


from classes import *


def Argos(opt):

   img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)

   device = torch_utils.select_device(force_cpu=ONNX_EXPORT)




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
   students=[]



   info=student(threadID,TKD_decoder,o_model,opt.source[0],opt,dist,device)

   # Configure run

   nc = 9  # number of classes

   seen = 0
   model.eval()
   coco91class = coco80_to_coco91_class()
   s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
   p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.

   jdict, stats, ap, ap_class = [], [], [], []



   gt = sio.loadmat('matlab.mat')
   gt = gt['bb']
   gt = gt[0]
   gt_counter = 0
   iou_thres = 0.5
   folder = 'aeroplane'

   for cl_counter in range(9):


       model.eval()

       info.TKD.eval().cuda()
       # Set Dataloader

       dataset = LoadImages('/media/common/DATAPART1/datasets/YouTube-Objects/videos/' + folder, img_size=info.opt.img_size, half=info.opt.half)

       # Get classes and colors
       classes = load_classes(parse_data_cfg(info.opt.data)['names'])

       if folder=='aeroplane':
           c_folder='airplane'
       elif folder=='motorbike':
           c_folder='motorcycle'
       else:
           c_folder=folder
       tcls_temp = classes.index(c_folder)


       # Run inference
       info.frame = torch.zeros([1, 3, info.opt.img_size, info.opt.img_size])
       oracle_T = Oracle()
       info.oracle.train().cuda()

       for path, img, im0s, vid_cap in dataset:

           info.collecting = True
           # Get detections

           image_index =path.split('/')[8].split('.')[0]

           info.frame[0, :, 0:img.shape[1], :] = torch.from_numpy(img)
           info.frame = info.frame.cuda()
           pred, _, feature = model(info.frame)
           info.TKD.img_size = info.frame.shape[-2:]
           pred_TKD, _ = info.TKD(feature)
           pred = torch.cat((pred, pred_TKD), 1)  # concat tkd and general decoder

           if not oracle_T.is_alive():
               oracle_T = Oracle()
               oracle_T.frame=info.frame
               oracle_T.feature=[Variable(feature[0].data, requires_grad=False),Variable(feature[1].data, requires_grad=False)]
               oracle_T.info=info
               oracle_T.start()

           # oracle_T.join()






           b = str(gt[gt_counter][0][0]).split('0', 1)
           if int(b[1]) == int(image_index):
               pred=non_max_suppression(pred, info.opt.conf_thres, info.opt.nms_thres)
               pred = pred[0]
               if pred is not None:
                   pred[:, :4] = scale_coords(img.shape[1:], pred[:, :4], im0s.shape).round()

               seen += 1

               labels=[]

               for j in gt[gt_counter][1]:

                   labels.append([tcls_temp,j[0],j[1],j[2],j[3]])


               labels=torch.FloatTensor(labels).cuda()
               gt_counter += 1
               b = str(gt[gt_counter][0][0]).split('0', 1)
               nl = len(labels)

               if pred is None:

                   if nl:
                       stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                   continue

               tcls = labels[:, 0].tolist() if nl else []  # target class
               correct = [0] * len(pred)

               if nl:
                   detected = []
                   tcls_tensor = labels[:, 0]

                   # target boxes

                   tbox = labels[:, 1:5]

                   # Search for correct predictions
                   for i, det in  enumerate(pred):

                       pbox=det[0:4]

                       pcls=det[6]

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
                       if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                           correct[i] = 1
                           detected.append(m[bi])
               # Append statistics (correct, conf, pcls, tcls)
               #print(correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls )
               stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))
               stats1 = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
               if len(stats1):
                   p, r, ap, f1, ap_class = ap_per_class(*stats1)
                   mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
               print(seen, mp, mr, map, mf1)


           if (b[0]) != folder:
               folder = b[0]
               break



                       # Stream results
                       # if stream_img:

           info.results = []



       stats1 = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
       if len(stats1):
           p, r, ap, f1, ap_class = ap_per_class(*stats1)
           mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
           #nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
       else:
           nt = torch.zeros(1)

       # Print results
       pf = '%20s' + '%10.3g' * 6  # print format
       #print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))
       print( seen, mp, mr, map, mf1)




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
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.3, help='iou threshold for non-maximum suppression')
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