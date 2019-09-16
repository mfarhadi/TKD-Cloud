import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *


def test(cfg,
         cfg_or,
         data,
         weights_or=None,
         weights=None,
         batch_size=16,
         img_size=416,
         iou_thres=0.5,
         conf_thres=0.001,
         nms_thres=0.5,
         save_json=False,
         model=None,
         oracle=None,
         ):

    if oracle is None:
        device = torch_utils.select_device()
        verbose = True

        # Initialize model
        oracle = Darknet(cfg_or, img_size).to(device)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            oracle.load_state_dict(torch.load(weights_or, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(oracle, weights_or)

        if torch.cuda.device_count() > 1:
            oracle = nn.DataParallel(oracle)
    else:
        device = next(oracle.parameters()).device  # get model device
        verbose = False

    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device()
        verbose = True

        # Initialize model
        model = Darknet(cfg, img_size).to(device)

        # Load weights
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:
        device = next(model.parameters()).device  # get model device
        verbose = False



    # Configure run
    data = parse_data_cfg(data)
    nc = int(data['classes'])  # number of classes
    test_path = data['valid']= 'data/coco/5k.txt'  # path to test images
    names = load_classes(data['names'])  # class names

    # Dataloader
    dataset = LoadImagesAndLabels(test_path, img_size, batch_size)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=min(os.cpu_count(), batch_size),
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    oracle.eval()
    coco91class = coco80_to_coco91_class()
    s = ('%30s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP', 'F1')
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3)
    jdict, stats, ap, ap_class = [], [], [], []
    correct_box = 0
    pred_box=0
    target_box=0
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        targets = targets.to(device)
        imgs = imgs.to(device)
        _, _, height, width = imgs.shape  # batch size, channels, height, width

        # Plot images with bounding boxes
        #if batch_i == 0 and not os.path.exists('test_batch0.jpg'):
        #    plot_images(imgs=imgs, targets=targets, paths=paths, fname='test_batch0.jpg')

        # Run model

        inf_out, train_out = model(imgs)  # inference and training outputs

        #inf_out_or, train_out_or = oracle(imgs)  # inference and training outputs

        # Compute loss
        if hasattr(model, 'hyp'):  # if model has loss hyperparameters
            loss += compute_loss(train_out, targets, model)[1][[0, 2, 3]].cpu()  # GIoU, obj, cls

        # Run NMS

        output = non_max_suppression(inf_out, conf_thres=conf_thres, nms_thres=nms_thres)

        # Statistics per image

        for si, pred in enumerate(output):

            labels = targets[targets[:, 0] == si, 1:]

            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append(([], torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for di, d in enumerate(pred):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(d[6])],
                                  'bbox': [floatn(x, 3) for x in box[di]],
                                  'score': floatn(d[4], 5)})

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Assign all predictions as incorrect
            correct = [0] * len(pred)

            if nl:
                detected = []


                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                tbox[:, [0, 2]] *= width
                tbox[:, [1, 3]] *= height

                # Search for correct predictions
                #print('----------------------')
                #print(pred[:][0:4])
                #print('---------1-----------')
                #print(pred[:][0:4])
                #print('----------------------')
                #print('---------2-----------')
                #print(pred)
                #print('----------------------')

                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):

                    tempe_img=imgs[si][:,int(pbox[1]):int(pbox[3]),int(pbox[0]):int(pbox[2])]
                    tempe_img=tempe_img[None,:,:,:]
                    #print(tempe_img.shape)
                    inf_out_or, train_out_or = oracle(tempe_img)
                    temp_out= non_max_suppression(inf_out_or, conf_thres=conf_thres, nms_thres=nms_thres)
                    print(temp_out)


                    #print(int(pbox[0]),int(pbox[2]),int(pbox[1]),int(pbox[3]),imgs.shape,tempe_img.shape)
                for tbox_T in tbox:
                    #print(tbox_T)
                    target_box+=1

                    iou_total=0
                    for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):
                        #print(pconf,pcls_conf)
                        iou = bbox_iou_modefied(pbox, tbox_T)
                        if iou>iou_total:
                            iou_total=iou



                    #iou_box, bi_box = bbox_iou(tbox_T, pred[:][0:4]).max(0)
                    if(iou_total>0.5):
                        correct_box +=1



                pred_box +=len(pred)
                for i, (*pbox, pconf, pcls_conf, pcls) in enumerate(pred):


                    # Break if all targets already located in image
                    if len(detected) == nl:
                        break

                    # Continue if predicted class not among image classes
                    if pcls.item() not in tcls:
                        continue

                    # Best iou, index between pred and targets
                    m = (pcls == tcls_tensor).nonzero().view(-1)

                    #print('---------3-----------')
                    #print(tbox[m])
                    #print('----------------------')

                    iou, bi = bbox_iou(pbox, tbox[m]).max(0)


                    # If iou > threshold and class is correct mark as correct
                    if iou > iou_thres and m[bi] not in detected:  # and pcls == tcls[bi]:
                        correct[i] = 1
                        detected.append(m[bi])


            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct, pred[:, 4].cpu(), pred[:, 6].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in list(zip(*stats))]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%30s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))
    print("recall Boxes:",'Confidence:',conf_thres,correct_box,seen, target_box, float(correct_box)/target_box, float(pred_box-correct_box)/target_box,pred_box-correct_box)

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Save JSON
    if save_json and map and len(jdict):
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO('../coco/annotations/instances_val2014.json')  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        map = cocoEval.stats[1]  # update mAP to pycocotools mAP

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--cfg_or', type=str, default='cfg/yolov3-spp.cfg', help='oracle cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3_spp.pt', help='path to weights file')
    parser.add_argument('--weights_or', type=str, default='weights/yolov3_spp.pt', help='path to oracle weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        results = test(opt.cfg,
                       opt.cfg_or,
                       opt.data,
                       opt.weights_or,
                       opt.weights,
                       opt.batch_size,
                       opt.img_size,
                       opt.iou_thres,
                       opt.conf_thres,
                       opt.nms_thres,
                       opt.save_json,
                       )
