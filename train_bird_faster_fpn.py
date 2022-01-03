import os
import numpy as np
import json
from detectron2.structures import BoxMode
import itertools
import cv2
from detectron2.data import MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import inference_on_dataset
from PIL import Image  
import PIL
from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances
import argparse
import glob

parser = argparse.ArgumentParser(description='image inference')
parser.add_argument('--path', '-p', default= './', help='maga images path')
parser.add_argument('--threshold', '-t', default= 0.5, help='confidence threshold')
parser.add_argument('--model', '-m', default= 'new_lbai_FPN', help='model name')
parser.add_argument('--weight', '-w', default= './', help='initial weight')
args = parser.parse_args()

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.data import build_detection_test_loader, build_detection_train_loader

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "D3")
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        # if cfg.MODEL.DENSEPOSE_ON:
        #     evaluators.append(DensePoseCOCOEvaluator(dataset_name, True, output_folder))
        return DatasetEvaluators(evaluators)
    # @classmethod
    # def build_test_loader(cls, cfg, dataset_name):
    #     return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))



def make_trainer(image_dir,model_name,image_num,lr):
    register_coco_instances("bird_dataset", {}, image_dir+"/tree.json", image_dir)
    birds_metadata = MetadataCatalog.get("train")
    birds_metadata.thing_classes = ['bird']
    cfg = get_cfg()
    cfg.merge_from_file("./pretrained_weight/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.OUTPUT_DIR = './models/'+model_name
    cfg.DATASETS.TRAIN = ("bird_dataset",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.STEPS = (30000,45000)
    cfg.SOLVER.GAMMA = 0.1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.SOLVER.MAX_ITER = 60000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg) 
    trainer.resume_or_load(resume=False)
    return trainer

trainer = make_trainer(args.path+'/all_small',args.model,len(glob.glob(args.path+'/small/*.JPG')+glob.glob(args.path+'/small/*.jpg')+glob.glob(args.path+'/small/*.png')),0.01)
trainer.train()