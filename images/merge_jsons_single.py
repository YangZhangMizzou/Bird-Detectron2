import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='image inference')
parser.add_argument('--path', '-p', default= '/home/yangzhang/coco/lBAI/val2017', help='images path')
parser.add_argument('--background', '-b', default= 'Yes', help='images path')
args = parser.parse_args()

bird_dictionary = {
	'Tree':1,
	'Bush':2
}
categories = []
annotations = []
images = []
types = {'type' : 'instances'}
for dic in bird_dictionary:
	categories.append({
	'supercategory' : 'none',
	'id' : bird_dictionary[dic] ,
	'name' :  dic
	})

file_dir = args.path
json_list = sorted(glob.glob(file_dir+'/*.json'))
image_list = sorted(glob.glob(file_dir+'/*.jpg'))+sorted(glob.glob(file_dir+'/*.JPG'))+sorted(glob.glob(file_dir+'/*.png'))
image_type = image_list[0].split('.')[-1]
image_dic = {}
image_id = 1
save_name = 'bird_real.json'
h,w,c = cv2.imread(image_list[0]).shape


if (args.background == 'Yes'):
	for image in image_list:
		image_dic[image_id] = image.split('/')[-1]
		image_id+=1
else:
	for json_dir in json_list:
		if (json_dir.split('/')[-1] != save_name):
			image_dic[image_id] = json_dir.split('/')[-1].replace('json',image_type)
			image_id+=1



for key,value in image_dic.items():
	info = {
		"id": key,
		"file_name": value,
		"width": w, 
		"height": h
		}
	images.append(info)

def get_bbox(data):
	x1 = data[0][0]
	x2 = data[1][0]
	y1 = data[0][1]
	y2 = data[1][1]
	return [min(x1,x2),min(y1,y2),max(x1,x2)-min(x1,x2),max(y1,y2)-min(y1,y2)],(max(x1,x2)-min(x1,x2)) * (max(y1,y2)-min(y1,y2))


annotations_id = 1
for json_dir in json_list:
	image_name = json_dir.split('/')[-1].split('.')[0]
	for key,value in image_dic.items():
		if value.split('.')[0] == image_name:
			image_name = value
			with open(json_dir,'r') as f:
				data = json.load(f)
			areas = []
			for item in data['shapes']:
				bbox,area =  get_bbox(item['points'])
				areas.append(area/1000.0)
				info =  {
					'id' : annotations_id,
					'category_id' : bird_dictionary[item['label']],
					'iscrowd' : 0,
					'ignore' : 0,
					'bbox' : bbox,
					'area' : area,
					'image_id' : key
				}       	
				annotations.append(info)
				annotations_id +=1
	
new_dict = {}
new_dict['images'] = images
new_dict['annotations'] = annotations
new_dict['type'] = types
new_dict['categories'] = categories
with open(file_dir+'/'+save_name,'w') as f:
	json.dump(new_dict,f,sort_keys = True,indent=4)





