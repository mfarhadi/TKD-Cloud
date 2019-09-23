import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn



def TKD_loss(output,target, loss):


    bs = output.shape[0]

    nc=output.shape[4]


    output = output.view(-1, nc)
    target = target.view(-1, nc)
    mask_output_z=torch.zeros(output.shape)
    mask_output_o = torch.ones(output.shape)

    det_confs = torch.sigmoid(target[:,4])


    confidence, confidence_id=det_confs.sort(descending=True)
    confidence_conter=0
    object_list=[]
    tmp_taget=(target.data * 0.4) + (output.data * 0.6)

    for i in range(confidence.shape[0]):
        if float(float(confidence[i].cpu().data))<0.4:
            break
        confidence_conter+=1

        mask_output_z[int(confidence_id[i].cpu().data),:]=1
        mask_output_o[int(confidence_id[i].cpu().data),:]=0
        tmp_taget[int(confidence_id[i].cpu().data),:]=target.data[int(confidence_id[i].cpu().data),:]
        object_list.append(int(confidence_id[i].cpu().data))


    ghamar=Variable(tmp_taget, requires_grad=False)
    Total_error=loss(output, ghamar)


    return Total_error

