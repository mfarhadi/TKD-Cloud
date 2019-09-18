import argparse
import time
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


def Fast_detection(model, info):
    model.eval()
    webcam = info.source == '0' or info.source.startswith('rtsp') or info.source.startswith('http')
    streams = info.source == 'streams.txt'

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
    for path, img, im0s, vid_cap in dataset:

      info.collecting=True
      # Get detections

      info.frame[0,:,0:img.shape[1],:] = torch.from_numpy(img)

      pred, p,feture = model(info.frame.cuda())
      print(p[0].shape, feture[0].shape)
      for i in feture:
          print(i.shape)

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







