from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import random
from detectron2.config import get_cfg
import os
import cv2
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from compare_json_import import compare
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='image inference')
parser.add_argument('--path', '-p', default= './', help='images path')
parser.add_argument('--threshold', '-t', default= 0.5, help='confidence threshold')
parser.add_argument('--model', '-m', default= 'new_lbai_FPN', help='model name')
args = parser.parse_args()
root_dir = args.path
threshold = float(args.threshold)

model_name = args.model
cfg = get_cfg()
cfg.merge_from_file('./pretrained_weight/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml')
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8], [16], [32],[64],[128]]
cfg.OUTPUT_DIR = model_name
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

def read_gps(image_name,info_dir):
    info_txt = open(info_dir,'r')
    for line in info_txt.readlines():
        img_name,_,coor_lat,coor_lont,altitude = line.split(' ')
        if (img_name in image_name):
            coor_lat = float(coor_lat.replace('(','').replace(',',''))
            coor_lont = float(coor_lont.replace('(','').replace(',',''))
            altitude = float(altitude.split(')')[0])
            return coor_lat, coor_lont, altitude

def read_gps_new(image_name):
  coor_lat = 39.761838833333336
  coor_lont = -93.23834030555555
  if image_name.split('m')[-1] == '.JPG':
    altitude = 208 + int(image_name.split('m')[0].split('_')[-1])
  else:
    altitude = 208 + 90
  return coor_lat, coor_lont, altitude

def get_sub_image(mega_image,image_name,overlap=0.2,ratio=1):
    #mage_image: original image
    #ratio: ratio * 512 counter the different heights of image taken
    #return: list of sub image and list fo the upper left corner of sub image
    coor_list = []
    sub_image_list = []
    w,h,c = mega_image.shape
    size  = int(ratio*512)
    num_rows = int(w/int(size*(1-overlap)))
    num_cols = int(h/int(size*(1-overlap)))
    new_size = int(size*(1-overlap))
    for i in range(num_rows+1):
        if (i == num_rows):
            for j in range(num_cols+1):
                if (j==num_cols):
                    sub_image = mega_image[-size:,-size:,:]
                    coor_list.append([w-size,h-size])
                    sub_image_list.append (sub_image)
                else:
                    sub_image = mega_image[-size:,new_size*j:new_size*j+size,:]
                    coor_list.append([w-size,new_size*j])
                    sub_image_list.append (sub_image)
        else:
            for j in range(num_cols+1):
                if (j==num_cols):
                    sub_image = mega_image[new_size*i:new_size*i+size,-size:,:]
                    coor_list.append([new_size*i,h-size])
                    sub_image_list.append (sub_image)
                else:
                    sub_image = mega_image[new_size*i:new_size*i+size,new_size*j:new_size*j+size,:]
                    coor_list.append([new_size*i,new_size*j])
                    sub_image_list.append (sub_image)
    return sub_image_list,coor_list

def get_msd(bbox_list,alpha=0.5):
    num_of_annos = len(bbox_list)
    total_size = 0
    for box in bbox_list:
        total_size += (box[2]-box[0])*(box[3]-box[1])
    average_size = total_size/num_of_annos
    new_list = []
    for box in bbox_list:
        init_msd = 0
        if ((box[2]-box[0])*(box[3]-box[1])-average_size) > 0:
            init_msd = 0.4*((box[2]-box[0])*(box[3]-box[1])-average_size)/average_size
        else:
            init_msd  = abs((box[2]-box[0])*(box[3]-box[1])-average_size)/average_size
        if init_msd > 1:
            init_msd = 1
        msd_score =  1 - init_msd
        box.append(alpha*msd_score+(1-alpha)*box[4])
        new_list.append(box)
    return np.asarray(new_list)

def py_cpu_nms(dets, thresh):  
    """Pure Python NMS baseline.""" 
    dets = np.asarray(dets) 
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 4] 
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = scores.argsort()[::-1]  
 
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        inds = np.where(ovr <= thresh)[0]  
        order = order[inds + 1]  
    return keep

def msd_nms(dets, thresh):  
    """Pure Python NMS baseline.""" 
    dets = np.asarray(dets) 
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 5]
    
  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  

    order = scores.argsort()[::-1]  
 
    keep = []  
    while order.size > 0:  
        i = order[0]  
        keep.append(i)  
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        inds = np.where(ovr <= thresh)[0]  
        order = order[inds + 1]  
    return keep

def size_filtering(annotations,threshold):
    num_of_annos = len(annotations)
    total_size = 0
    for annotation in annotations:
        total_size += annotation['size']
    average_size = total_size/num_of_annos
    for annotation in annotations:
        init_msd = 0
        if (annotation['size']-average_size) > 0:
            init_msd = 0.4*(annotation['size']-average_size)/average_size
        else:
            init_msd  =  abs(annotation['size']-average_size)/average_size
        if init_msd > 1:
            init_msd = 1
        annotation['area_bias'] = init_msd
    new_annotations = []
    for annotation in annotations:
        if annotation['area_bias'] < threshold:
            new_annotations.append(annotation)
    return new_annotations


import json
import os
import glob
import numpy as np 
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])


FOV =77#need calibration
height = 30
image_list = sorted(glob.glob(root_dir+'/*.JPG'))+sorted(glob.glob(root_dir+'/*.jpg'))+sorted(glob.glob(root_dir+'/*.png'))
mega_imgae_id = 0
bbox_id = 1
all_annotations= []

save_dir = root_dir+'/'+model_name
if (os.path.isdir(save_dir) == False):
    os.makedirs(save_dir)
print('There are '+str(len(image_list))+' images totally.')
with tqdm(total = len(image_list)) as pbar:
    for image_dir in image_list:
        pbar.update(1)
        bbox_list = []
        mega_imgae_id += 1
        mega_image  = cv2.imread(image_dir)
        ratio = 1.0
        sub_image_list,coor_list = get_sub_image(mega_image,image_dir.split('/')[-1],overlap = 0.1,ratio = ratio)
        for index,sub_image in enumerate(sub_image_list):
            inputs = cv2.resize(sub_image,(512,512),interpolation = cv2.INTER_AREA)
            outputs = predictor(inputs)
            boxes = outputs["instances"].to("cpu").get_fields()['pred_boxes'].tensor.numpy()
            score = outputs["instances"].to("cpu").get_fields()['scores'].numpy()
            labels = outputs["instances"].to("cpu").get_fields()['pred_classes'].numpy()
            if (len(boxes.shape)!=0):
                for idx in range(boxes.shape[0]):
                  x1,y1,x2,y2 = boxes[idx][0], boxes[idx][1] ,boxes[idx][2] ,boxes[idx][3]  # (x1,y1, x2,y2)
                  bbox_list.append([coor_list[index][1]+ratio*x1, coor_list[index][0]+ratio*y1, coor_list[index][1]+ratio*x2, coor_list[index][0]+ratio*y2,score[idx],labels[idx]])
        if (len(bbox_list)!=0):
            annotations = []
            # bbox_list = get_msd(bbox_list,0.5)#alpha, weight of msd
            # box_idx = msd_nms(bbox_list,0.2)#iou_thresh
            bbox_list = np.asarray(bbox_list)
            box_idx = py_cpu_nms(np.asarray(bbox_list),0.2)
            for box in bbox_list[box_idx]:
              tmp_dict = {
                'category_id' : int(box[-1] + 1) ,
                'bbox' : [int(box[0]),int(box[1]),int(box[2])-int(box[0]),int(box[3])-int(box[1])],
                'score' : box[4],
                'confidence': box[4],
                'size'  : (int(box[2])-int(box[0]))*(int(box[3])-int(box[1])),
                'image_id' : mega_imgae_id 
              }
              annotations.append(tmp_dict)
              bbox_id += 1
            # annotations = size_filtering(annotations,0.7)
            for annotation in annotations:
                if annotation['score']>0.5:
                    box = annotation['bbox']
                    confidnece = annotation['confidence']
                    cv2.putText(mega_image, str(round(confidnece,2)), (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if annotation['category_id'] == 1:
                        cv2.rectangle(mega_image, (int(box[0]),int(box[1])),(int(box[0])+int(box[2]),int(box[1])+int(box[3])), (255,0,0), 2)
                    else:
                         cv2.rectangle(mega_image, (int(box[0]),int(box[1])),(int(box[0])+int(box[2]),int(box[1])+int(box[3])), (0,0,255), 2)

            all_annotations.extend(annotations)
        cv2.imwrite (save_dir+'/'+image_dir.split('/')[-1],mega_image)


with open(save_dir+'/bird.json','w') as f:
    json.dump(all_annotations,f,sort_keys = True,indent=4)

if os.path.exists(root_dir+'/bird.json'):
    compare(save_dir+'/bird.json',root_dir+'/bird.json',save_dir+'threshold=_'+str(threshold)+'.txt',save_dir+'threshold=_'+str(threshold)+'.csv',float(threshold))
