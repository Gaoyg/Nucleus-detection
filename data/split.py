import os
import shutil
import random

root_path = '/home/bio-eecs/gyg/nucleus_detection/data/'
input_path = root_path + 'stage1_train/'

ids = []

for index in os.listdir(input_path):
	ids.append(index)


random.shuffle(ids)

length = len(ids)
split_point = length*4/5
train_ids = ids[:split_point]
val_ids = ids[split_point:]

for index in train_ids:
	shutil.copytree(input_path+index, root_path+'train/'+index)


for index in val_ids:
	shutil.copytree(input_path+index, root_path+'val/'+index)