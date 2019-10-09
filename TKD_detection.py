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



import threading

class Oracle(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.frame=None
        self.feature=None
        self.info=None
    def run(self):

        Retraining(self.frame, self.feature, self.info)



def receive_tensor_helper(dist,tensor, src_rank, group, tag, num_iterations,
                          intra_server_broadcast):
    for i in range(num_iterations):
        if intra_server_broadcast:
            dist.broadcast(tensor=tensor, group=group, src=src_rank)
        else:
            dist.recv(tensor=tensor, src=src_rank, tag=tag)


def send_tensor_helper(dist,tensor, dst_rank, group, tag, num_iterations,
                       intra_server_broadcast):
    for i in range(num_iterations):
        if intra_server_broadcast:
            dist.broadcast(tensor=tensor, group=group, src=1-dst_rank)
        else:
            dist.send(tensor=tensor, dst=dst_rank, tag=tag)





def Fast_detection(model, info):
    model.eval()
    webcam = info.source == '0' or info.source.startswith('rtsp') or info.source.startswith('http')
    streams = info.source == 'streams.txt'
    info.TKD.eval().cuda()
    # Set Dataloader
    vid_path, vid_writer = None, None
    if streams:
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(info.source, img_size=info.opt.img_size, half=info.opt.half)
    elif webcam:
        stream_img = True
        dataset = LoadWebcam(info.source, img_size=info.opt.img_size, half=info.opt.half)
    else:
        save_img = True
        print('salam')
        dataset = LoadImages(info.source, img_size=info.opt.img_size, half=info.opt.half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(info.opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    info.frame = torch.zeros([1, 3, info.opt.img_size, info.opt.img_size])
    oracle_T=Oracle()
    info.oracle.train().cuda()

    for path, img, im0s, vid_cap in dataset:

      info.collecting=True
      # Get detections


      info.frame[0, :, 0:img.shape[1], :] = torch.from_numpy(img)
      info.frame = info.frame.cuda()
      pred, _, feature = model(info.frame)
      info.TKD.img_size = info.frame.shape[-2:]
      pred_TKD, _ = info.TKD(feature)
      pred = torch.cat((pred, pred_TKD), 1)  # concat tkd and general decoder

      #test_v=non_max_suppression(pred, info.opt.conf_thres, info.opt.nms_thres)
      #print(test_v[0])

      if not oracle_T.is_alive():
          oracle_T = Oracle()
          oracle_T.frame=info.frame
          oracle_T.feature=[Variable(feature[0].data, requires_grad=False),Variable(feature[1].data, requires_grad=False)]
          oracle_T.info=info
          oracle_T.start()

      #oracle_T.join()


      for i, det in enumerate(non_max_suppression(pred, info.opt.conf_thres, info.opt.nms_thres)):  # detections per image
          s,im0='',im0s


          if det is not None and len(det):

              # Rescale boxes from img_size to im0 size

              det[:, :4] = scale_coords(img.shape[1:], det[:, :4], im0.shape).round()

              # Print results
              for c in det[:, -1].unique():
                  n = (det[:, -1] == c).sum()  # detections per class
                  s += '%g %ss, ' % (n, classes[int(c)])  # add to string

              # Write results
              for *xyxy, conf, _, cls in det:
                  label = '%s %.2f' % (classes[int(cls)], conf)
                  plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])



          # Stream results
          # if stream_img:

      info.results=[]

      cv2.imshow(str(info.threadID), im0s)
      cv2.waitKey(1)


      if info.exitFlag:
          break




def Retraining(frame, feature,info):

    if info.network:
        tensor = frame.cpu()
        send_tensor_helper(info.dist, tensor, 1 - info.opt.rank, 0, 0,
                           1, info.opt.intra_server_broadcast)
        for parm in info.TKD.parameters():
            temp_w=parm.cpu()
            receive_tensor_helper(info.dist, temp_w, 1 - info.opt.rank, 0, 0,
                                  1, info.opt.intra_server_broadcast)
            parm[:] = temp_w.cuda()
    else:
        T_out = info.oracle(frame)
        richOutput=[Variable(T_out[0].data, requires_grad=False),Variable(T_out[1].data, requires_grad=False)]


        for j in range(3):

            info.optimizer.zero_grad()

            S_out, p = info.TKD(feature)

            loss = 0

            for i in range(2):

                loss += TKD_loss(p[i],richOutput[i],info.loss)
            loss.backward(retain_graph=True)
            info.optimizer.step()

        print("TKD Loss",loss.data.cpu())








def server_Retraining(info):
    args=info.opt
    num_ranks_in_server = 1
    if args.intra_server_broadcast:
        num_ranks_in_server = 2
    local_rank = args.rank % num_ranks_in_server
    torch.cuda.set_device(local_rank)

    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    world_size = 2
    dist.init_process_group(args.backend, rank=args.rank, world_size=world_size)
    print('initied')
    info.oracle.train()
    info.TKD=info.TKD.cuda().eval()
    while not info.exitFlag:

        tensor = torch.zeros(([1, 3, 416, 416])).cpu()

        receive_tensor_helper(dist,tensor, 1-args.rank, 0, 0,
                             1, args.intra_server_broadcast)
        tensor=tensor.cuda()

        info.TKD.img_size = tensor.shape[-2:]
        T_out = info.oracle(tensor)

        pred, _, feature = info.model(tensor)
        richOutput = [Variable(T_out[0].data, requires_grad=False), Variable(T_out[1].data, requires_grad=False)]
        feature = [Variable(feature[0].data, requires_grad=False), Variable(feature[1].data, requires_grad=False)]

        for j in range(3):

            info.optimizer.zero_grad()
            S_out, p = info.TKD(feature)
            loss = 0

            for i in range(2):

                loss += TKD_loss(p[i], richOutput[i], info.loss)
            loss.backward(retain_graph=True)
            info.optimizer.step()

        for parm in info.TKD.parameters():
            send_tensor_helper(dist, parm.cpu(), 1 - info.opt.rank, 0, 0,
                               1, info.opt.intra_server_broadcast)

        print("TKD Loss", loss.data.cpu())










