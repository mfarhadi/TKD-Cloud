import argparse
import time
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def Fast_detection(model, students):
    model.eval()
    while students:
        input=[]
        list=[]

        for student in students:

            if student.collecting==False:
                input.append(student.frame)
                list.append(student)

        if input:
            input=torch.cat(input, dim=0).cuda()
            pred, _ = model(input)

            for i,student in enumerate(list):
                student.results=pred[i]
                student.collecting = True
                student.ready = True


