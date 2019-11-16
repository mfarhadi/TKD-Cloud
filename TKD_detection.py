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
from classes import *
from motion_detection import *

import socket
import numpy
import time
import random

try:
    import cPickle as pickle
except ImportError:
    import pickle



import threading

class frame_listen(threading.Thread):

    def __init__(self, student):
        threading.Thread.__init__(self)
        self.student=student
    def run(self):
        self.student.frame=[]
        args = self.student.opt
        while True:
            half = torch.ones(([1])).cpu()

            receive_tensor_helper(dist, half, 1 - args.rank, 0, 0,
                                  1, args.intra_server_broadcast)

            tensor = torch.zeros(([1, 3, 416, 416])).cpu()
            self.student.opt.half=half
            if half == 1:
                tensor = tensor.half()

            receive_tensor_helper(dist, tensor, 1 - args.rank, 0, 0,
                                  1, args.intra_server_broadcast)
            tensor = tensor.type(torch.FloatTensor)
            self.student.frame.append(tensor)


            self.student.TKD.img_size = tensor.shape[-2:]
            if len(self.student.frame)>9:
                del self.student.frame[0]
                time.sleep(0.01)

class weights_send(threading.Thread):

    def __init__(self, student,loss):
        threading.Thread.__init__(self)
        self.student=student
        self.loss=loss
    def run(self):

        for parm in self.student.TKD.parameters():
            if self.student.opt.half == 1:
                send_tensor_helper(dist, parm.cpu().half(), 1 - self.student.opt.rank, 0, 0,
                                   1, self.student.opt.intra_server_broadcast)
            else:
                send_tensor_helper(dist, parm.cpu(), 1 - self.student.opt.rank, 0, 0,
                                   1, self.student.opt.intra_server_broadcast)

        send_tensor_helper(dist, self.loss.cpu(), 1 - self.student.opt.rank, 0, 0,
                           1, self.student.opt.intra_server_broadcast)

class Remote_precision(threading.Thread):

    def __init__(self, image,detection, info):
        threading.Thread.__init__(self)
        self.image=image
        if detection is None:
            self.result=np.zeros(1)
        else:
            self.result=detection.detach().cpu().numpy()
        self.info=info
        self.socket=info.socket

    def run(self):
        #print(self.image.type(),self.result.type())
        print(self.info.loss.cpu())
        j = pickle.dumps(self.image, protocol=2)
        self.socket.sendall(str(len(j)).encode())
        data = self.socket.recv(1024)
        self.socket.sendall(j)
        data = self.socket.recv(1024)

        j = pickle.dumps(self.result, protocol=2)

        self.socket.sendall(str(len(j)).encode())

        data = self.socket.recv(1024)

        self.socket.sendall(j)
        data = self.socket.recv(1024)

        data=numpy.array([float(self.info.loss.cpu())])
        j = pickle.dumps(data, protocol=2)

        self.socket.sendall(str(len(j)).encode())

        data = self.socket.recv(1024)

        self.socket.sendall(j)
        data = self.socket.recv(1024)






class Oracle(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.frame=None
        self.feature=None
        self.info=None
    def run(self):

        Retraining(self.frame, self.feature, self.info)


class Oracle_listener(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.frame=None
        self.feature=None
        self.info=None
    def run(self):
        while True:
            print('ready to get results')
            for parm in self.info.TKD.parameters():
                temp_w = parm.cpu()
                receive_tensor_helper(self.info.dist, temp_w, 1 - self.info.opt.rank, 0, 0,
                                      1, self.info.opt.intra_server_broadcast)
                parm[:] = temp_w.cuda()

            loss = torch.zeros(([1])).cpu()

            receive_tensor_helper(self.info.dist, loss, 1 - self.info.opt.rank, 0, 0,
                                  1, self.info.opt.intra_server_broadcast)

            if float(loss.data.cpu()) < float(self.info.loss.data.cpu()) - 0.1 or float(loss.data.cpu()) > float(
                    self.info.loss.data.cpu()) + 1:
                if self.info.threshold < 50:
                    self.info.threshold *= 2
                elif self.info.threshold < 90:
                    self.info.threshold += 10
            else:
                if self.info.threshold > 10:
                    self.info.threshold -= 1

            self.info.loss = loss
            print("TKD Loss", loss.data.cpu())



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

        dataset = LoadImages(info.source, img_size=info.opt.img_size, half=info.opt.half)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(info.opt.data)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    # Run inference
    info.frame = torch.zeros([1, 3, info.opt.img_size, info.opt.img_size])
    oracle_T=Oracle()

    if info.opt.ctraining and info.network:
        oracle_re=Oracle_listener()
        oracle_re.info=info
        oracle_re.start()

    rem_prec = Remote_precision(info.frame, info.frame, info)
    info.oracle.train().cuda()

    m_info=motion_info()
    m_detect=motion_detection(info.frame,m_info)

    for path, img, im0s, vid_cap in dataset:

      #motion detection

      if not m_detect.is_alive():
          m_detect = motion_detection(im0s, m_info)
          m_detect.start()


      info.collecting=True
      # Get detections


      info.frame[0, :, 0:img.shape[1], :] = torch.from_numpy(img)
      if info.opt.half:
        info.frame = info.frame.half().cuda()
      else:
        info.frame = info.frame.cuda()
      inf_t1 = time.time()
      pred, _, feature = model(info.frame)
      info.TKD.img_size = info.frame.shape[-2:]
      pred_TKD, TKD_tensor = info.TKD(feature)
      pred = torch.cat((pred, pred_TKD), 1)  # concat tkd and general decoder

      inf_t2 = time.time()
      inf_file = open('inference_time' + '.txt', 'a')
      if info.network:
          inf_file.write('\n' + 'Network' + ',' + str(inf_t2 - inf_t1))
      else:
          inf_file.write('\n' + 'Local' + ',' + str(inf_t2 - inf_t1))
      inf_file.close()

      #test_v=non_max_suppression(pred, info.opt.conf_thres, info.opt.nms_thres)
      #print(test_v[0])
      if m_info.motion_status or info.threshold>10:
          rand_value=random.uniform(0,100)
          if not oracle_T.is_alive() and rand_value<info.threshold:

              m_info.static_back=m_info.gray

              oracle_T = Oracle()
              oracle_T.frame=info.frame
              oracle_T.feature=[]
              for i3 in range(len(feature)):
                  oracle_T.feature.append(Variable(feature[i3].data, requires_grad=False))
              oracle_T.info=info
              oracle_T.start()
              print('selected', rand_value, info.threshold)
          else:
              if rand_value>info.threshold:
                  print('Not selected',rand_value,info.threshold)

      #oracle_T.join()
      detection=non_max_suppression(pred, info.opt.conf_thres, info.opt.nms_thres)

      if not rem_prec.is_alive() and info.precision:
          rem_prec=Remote_precision(info.frame.cpu().numpy(),detection[0],info)
          rem_prec.start()

      for i, det in enumerate(detection):  # detections per image
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
    t1=time.time()
    if info.network:
        tensor = frame.cpu()
        if info.opt.half:
            send_tensor_helper(info.dist,torch.ones(([1])).cpu(), 1 - info.opt.rank, 0, 0,
                           1, info.opt.intra_server_broadcast)
        else:
            send_tensor_helper(info.dist,torch.zeros(([1])).cpu(), 1 - info.opt.rank, 0, 0,
                           1, info.opt.intra_server_broadcast)

        send_tensor_helper(info.dist, tensor, 1 - info.opt.rank, 0, 0,
                           1, info.opt.intra_server_broadcast)
        if not info.opt.ctraining:
            for parm in info.TKD.parameters():
                temp_w=parm.cpu()
                receive_tensor_helper(info.dist, temp_w, 1 - info.opt.rank, 0, 0,
                                      1, info.opt.intra_server_broadcast)
                parm[:] = temp_w.cuda()

            loss = torch.zeros(([1])).cpu()

            receive_tensor_helper(info.dist, loss, 1 - info.opt.rank, 0, 0,
                                  1, info.opt.intra_server_broadcast)

    else:
        T_out = info.oracle(frame)

        richOutput=[]
        for i3 in range(len(T_out)):
            richOutput.append(Variable(T_out[i3].data, requires_grad=False))


        for j in range(3):

            info.optimizer.zero_grad()

            S_out, p = info.TKD(feature)

            loss = 0

            for i in range(len(p)):

                loss += TKD_loss(p[i],richOutput[i],info.loss_F)
            if loss > 0.3:
                loss.backward(retain_graph=True)
                info.optimizer.step()


    t2=time.time()
    if info.network and info.opt.ctraining:
        file = open('sendimage_time' + '.txt', 'a')
    else:

        if float(loss.data.cpu()) < float(info.loss.data.cpu()) - 0.1 or float(loss.data.cpu()) > float(
                info.loss.data.cpu()) + 1:
            if info.threshold < 50:
                info.threshold *= 2
            elif info.threshold < 90:
                info.threshold += 10
        else:
            if info.threshold > 10:
                info.threshold -= 1

        info.loss = loss
        print("TKD Loss", loss.data.cpu())
        file = open('time' + '.txt', 'a')
    if info.network:
        file.write('\n' +'Network'+','+ str(t2-t1))
    else:
        file.write('\n' + 'Local' + ',' + str(t2 - t1))
    file.close()










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
    print('start')
    dist.init_process_group(args.backend, rank=args.rank, world_size=world_size)
    print('initied')
    info.oracle.train()
    info.TKD=info.TKD.cuda().eval()
    reciver=frame_listen(info)
    reciver.start()
    w_sender=weights_send(1,1)
    while not info.exitFlag:
        '''
        half=torch.ones(([1])).cpu()

        receive_tensor_helper(dist,half, 1-args.rank, 0, 0,
                             1, args.intra_server_broadcast)

        tensor = torch.zeros(([1, 3, 416, 416])).cpu()

        if half==1:
            tensor=tensor.half()

        receive_tensor_helper(dist,tensor, 1-args.rank, 0, 0,
                             1, args.intra_server_broadcast)
        tensor=tensor.cuda().type(torch.cuda.FloatTensor)
        if info.opt.half:
            tensor=tensor.half()
        
        info.TKD.img_size = tensor.shape[-2:]
        '''
        half=info.opt.half
        if info.frame:

            tensor=torch.cat(info.frame, 0)
            del info.frame[:]
            tensor=tensor.cuda()
            T_out = info.oracle(tensor)

            pred, _, feature = info.model(tensor)
            del tensor
            torch.cuda.empty_cache()
            richOutput=[]
            for i3 in range(len(T_out)):
                richOutput.append(Variable(T_out[i3].data, requires_grad=False))

            for i3 in range(len(feature)):
                feature[i3]=Variable(feature[i3].data, requires_grad=False)


            for j in range(3):

                info.optimizer.zero_grad()
                S_out, p = info.TKD(feature)
                loss = 0

                for i in range(len(p)):

                    loss += TKD_loss(p[i], richOutput[i], info.loss)

                if loss>0.3:
                    loss.backward(retain_graph=True)
                    info.optimizer.step()
            if not w_sender.is_alive():
                w_sender=weights_send(info,loss)
                w_sender.run()



            #print("TKD Loss", loss.data.cpu())
        else:
            time.sleep(0.005)










