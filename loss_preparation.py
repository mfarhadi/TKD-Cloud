import sys
import os
import time
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
import torch.nn as nn
from utils2.utils import build_targets

import itertools
import struct # get_image_size
import imghdr # get_image_size

def sigmoid(x):
    return 1.0/(math.exp(-x)+1.)

def softmax(x):
    x = torch.exp(x - torch.max(x))
    x = x/x.sum()
    return x

def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)

def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)

def Tensor_edit(output,target, num_classes, num_anchors,loss, anchors=[]):
    #anchor_step = len(anchors)/num_anchors

    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (5+num_classes)*num_anchors)
    h = output.size(2)
    w = output.size(3)

    t0 = time.time()
    all_boxes = []

    output = output.view(batch*num_anchors, 5+num_classes, h*w).transpose(0,1).contiguous().view(5+num_classes, batch*num_anchors*h*w)
    target = target.view(batch * num_anchors, 5 + num_classes, h * w).transpose(0, 1).contiguous().view(5 + num_classes,batch * num_anchors * h * w)
    mask_output_z=torch.zeros(output.shape)
    mask_output_o = torch.ones(output.shape)

    det_confs = torch.sigmoid(target[4])

    '''
    grid_x = torch.linspace(0, w-1, w).repeat(h,1).repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).type_as(output) #cuda()
    grid_y = torch.linspace(0, h-1, h).repeat(w,1).t().repeat(batch*num_anchors, 1, 1).view(batch*num_anchors*h*w).type_as(output) #cuda()
    xs = torch.sigmoid(output[0]) + grid_x
    ys = torch.sigmoid(output[1]) + grid_y

    anchor_w = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([0]))
    anchor_h = torch.Tensor(anchors).view(num_anchors, anchor_step).index_select(1, torch.LongTensor([1]))
    anchor_w = anchor_w.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).type_as(output) #cuda()
    anchor_h = anchor_h.repeat(batch, 1).repeat(1, 1, h*w).view(batch*num_anchors*h*w).type_as(output) #cuda()
    ws = torch.exp(output[2]) * anchor_w
    hs = torch.exp(output[3]) * anchor_h

    det_confs = torch.sigmoid(output[4])
    #print(det_confs.sort(descending=True)[0][0])

    cls_confs = torch.nn.Softmax()(Variable(output[5:5+num_classes].transpose(0,1))).data

    cls_max_confs, cls_max_ids = torch.max(cls_confs, 1)
    cls_max_confs = cls_max_confs.view(-1)
    cls_max_ids = cls_max_ids.view(-1)
    t1 = time.time()
    
    sz_hw = h*w
    sz_hwa = sz_hw*num_anchors
    det_confs = convert2cpu(det_confs)
    cls_max_confs = convert2cpu(cls_max_confs)
    cls_max_ids = convert2cpu_long(cls_max_ids)
    xs = convert2cpu(xs)
    ys = convert2cpu(ys)
    ws = convert2cpu(ws)
    hs = convert2cpu(hs)

    if validation:
        cls_confs = convert2cpu(cls_confs.view(-1, num_classes))
    t2 = time.time()
    for b in range(batch):
        boxes = []
        for cy in range(h):
            for cx in range(w):
                for i in range(num_anchors):
                    ind = b*sz_hwa + i*sz_hw + cy*w + cx
                    det_conf =  det_confs[ind]
                    if only_objectness:
                        conf =  det_confs[ind]
                    else:
                        conf = det_confs[ind] * cls_max_confs[ind]
    
                    if conf > conf_thresh:
                        bcx = xs[ind]
                        bcy = ys[ind]
                        bw = ws[ind]
                        bh = hs[ind]
                        cls_max_conf = cls_max_confs[ind]
                        cls_max_id = cls_max_ids[ind]

                        box = [bcx/w, bcy/h, bw/w, bh/h, det_conf, cls_max_conf, cls_max_id]
                        if (not only_objectness) and validation:
                            for c in range(num_classes):
                                tmp_conf = cls_confs[ind][c]
                                if c != cls_max_id and det_confs[ind]*tmp_conf > conf_thresh:
                                    box.append(tmp_conf)
                                    box.append(c)
                        boxes.append(box)
        all_boxes.append(boxes)
    t3 = time.time()
    if False:
        print('---------------------------------')
        print('matrix computation : %f' % (t1-t0))
        print('        gpu to cpu : %f' % (t2-t1))
        print('      boxes filter : %f' % (t3-t2))
        print('---------------------------------')

    '''
    #target=target.cpu()
    confidence, confidence_id=det_confs.sort(descending=True)
    confidence_conter=0
    object_list=[]
    tmp_taget=(target.data * 0.4) + (output.data * 0.6)
    #tmp_confidence=output.data[4]
    #tmp_taget[4] = (target[4].data * 0.5) + (output[4].data * 0.5)
    #print("(target[4].data * 0.5) + (output[4].data * 0.5)")
    #tmp_taget[4]=(target.data[4] * 0.6) + (output.data[4] * 0.4)
    #print("0.75")
    for i in range(confidence.shape[0]):
        if float(float(confidence[i].cpu().data))<0.5:
            break
        confidence_conter+=1
        #tmp_confidence[int(confidence_id[i].cpu().data)]=target.data[4:int(confidence_id[i].cpu().data)]
        mask_output_z[:,int(confidence_id[i].cpu().data)]=1
        mask_output_o[:,int(confidence_id[i].cpu().data)]=0
        tmp_taget[:,int(confidence_id[i].cpu().data)]=target.data[:,int(confidence_id[i].cpu().data)]
        object_list.append(int(confidence_id[i].cpu().data))
    '''
        if confidence_conter==1:
            Total_error=loss(output[:,int(confidence_id[i].cpu().data)],target[:,int(confidence_id[i].cpu().data)])

        else:
            Total_error+=loss(output[:,int(confidence_id[i].cpu().data)],target[:,int(confidence_id[i].cpu().data)])

    if confidence_conter==0:
        Total_error=loss(output[4],target[4])
    else:
        Total_error += nn.MSELoss(size_average=False)(output[4], target[4])
    '''
    mask_output_o=mask_output_o.cuda()
    mask_output_z=mask_output_z.cuda()
    #tmp_taget= target.data*mask_output_z + output.data*mask_output_o
    #tmp_taget = (target.data + output.data)/2
    #tmp_taget[4]=target.data[4]

    #tmp_taget=target.data
    ghamar=Variable(tmp_taget, requires_grad=False)
    Total_error=loss(output, ghamar)
    #print(Result)

    return Total_error

class YOLO_loss2(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim):
        super(YOLO_loss2, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1

        self.mse_loss = nn.MSELoss(size_average=True)  # Coordinate loss
        self.bce_loss = nn.BCELoss(size_average=True)  # Confidence loss
        self.ce_loss = nn.CrossEntropyLoss()  # Class loss

    def forward(self, x, targets=None):
        #print(x.shape)
        #print(self.anchors)
        #print(targets)
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # Calculate offsets for each grid
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)

        grid_y = torch.t(torch.arange(nG).repeat(nG, 1).type(FloatTensor))
        grid_y=grid_y[None,None,:,:]

        scaled_anchors = FloatTensor([(float(a_w) / stride, float(a_h) / stride) for a_w, a_h in self.anchors])

        anchor_w = scaled_anchors[:, 0:1]
        anchor_w=anchor_w[None,:,None]


        anchor_h = scaled_anchors[:, 1:2]
        anchor_h = anchor_h[None, :, None]

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # Training
        if targets is not None:

            if x.is_cuda:
                self.mse_loss = self.mse_loss.cuda()
                self.bce_loss = self.bce_loss.cuda()
                self.ce_loss = self.ce_loss.cuda()

            nGT, nCorrect, mask, conf_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                pred_boxes=pred_boxes.cpu(),
                pred_conf=pred_conf.cpu(),
                pred_cls=pred_cls.cpu(),
                target=targets.cpu(),
                anchors=scaled_anchors.cpu(),
                num_anchors=nA,
                num_classes=self.num_classes,
                grid_size=nG,
                ignore_thres=self.ignore_thres,
                img_dim=self.image_dim,
            )

            #nProposals = int((pred_conf > 0.5).sum().item())
            #recall = float(nCorrect / nGT) if nGT else 1
            #precision = float(nCorrect / nProposals)

            # Handle masks
            mask = Variable(mask.type(ByteTensor))
            conf_mask = Variable(conf_mask.type(ByteTensor))

            # Handle target variables
            tx = Variable(tx.type(FloatTensor), requires_grad=False)
            ty = Variable(ty.type(FloatTensor), requires_grad=False)
            tw = Variable(tw.type(FloatTensor), requires_grad=False)
            th = Variable(th.type(FloatTensor), requires_grad=False)
            tconf = Variable(tconf.type(FloatTensor), requires_grad=False)
            tcls = Variable(tcls.type(LongTensor), requires_grad=False)

            # Get conf mask where gt and where there is no gt
            conf_mask_true = mask
            conf_mask_false = conf_mask - mask

            # Mask outputs to ignore non-existing objects
            loss_x = self.mse_loss(x[mask], tx[mask])
            loss_y = self.mse_loss(y[mask], ty[mask])
            loss_w = self.mse_loss(w[mask], tw[mask])
            loss_h = self.mse_loss(h[mask], th[mask])
            loss_conf = self.bce_loss(pred_conf[conf_mask_false], tconf[conf_mask_false]) + self.bce_loss(
                pred_conf[conf_mask_true], tconf[conf_mask_true]
            )
            #loss_cls = (1 / nB) * self.ce_loss(pred_cls[mask], torch.argmax(tcls[mask], 1))
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf# + loss_cls

            return (
                loss#,
                #loss_x,#.item(),
                #loss_y,#.item(),
                #loss_w,#.item(),
                #loss_h,#.item(),
                #loss_conf,#.item()#,
                #loss_cls.item(),
                #recall,
                #precision,
            )

        else:
            # If not in training phase return predictions
            output = torch.cat(
                (
                    pred_boxes.view(nB, -1, 4) * stride,
                    pred_conf.view(nB, -1, 1),
                    pred_cls.view(nB, -1, self.num_classes),
                ),
                -1,
            )
            return output

