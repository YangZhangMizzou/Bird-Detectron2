import json
import os
import glob

image_dir = '/home/yangzhang/detectron/datasets/Union_Data/images_to_train/test/self_rectification/PershingSP_Photos_self_lr_decay_0.5_round3/test'
json_dir = image_dir + '/bird.json'
with open(json_dir,'r') as f:
	data = json.load(f)
image_list = sorted(glob.glob(image_dir+'/*.JPG'))
image_name_list = []
image_name_dic = {}
id_index = 1
for image in image_list:
	image_name_list.append(image.split('/')[-1])
	image_name_dic[image.split('/')[-1]] = id_index
	id_index+=1
for image_name in image_name_list:
	with open('{}/{}.json'.format(image_dir,image_name.split('.')[0]),'r') as f:
		mega_data = json.load(f)
	mega_data['shapes']=[]
	for item in data:
		if item['image_id'] == image_name_dic[image_name]:
			item_new = {}
			item_new['points'] = [[item['bbox'][0],item['bbox'][1]],[item['bbox'][0]+item['bbox'][2],item['bbox'][1]+item['bbox'][3]]]
			item_new['shape_type'] = 'rectangle'
			item_new['group_id'] = None
			item_new['label'] = 'bird'
			item_new['flags'] = {}
			mega_data['shapes'].append(item_new)
	if mega_data['shapes']!=[]:
		with open(image_dir+'/'+image_name.split('.')[0]+'.json','w') as f:
			json.dump(mega_data,f)



