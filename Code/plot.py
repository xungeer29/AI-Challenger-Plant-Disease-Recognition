# -*- coding:utf-8 -*-

import json
from matplotlib import pyplot as plt
import numpy as np

train_label_dir = '/media/gfx/GFX/DATASET/Plant_Disease_Recognition/ai_challenger_pdr2018_train_annotations_20181021.json'
val_label_dir = '/media/gfx/GFX/DATASET/Plant_Disease_Recognition/ai_challenger_pdr2018_validation_annotations_20181021.json'
save_dir = './'

classes = 61

# trainset
with open(train_label_dir, 'r') as f_train:
    image_label_list_train = json.load(f_train)

num_per_label_train = np.zeros(classes, dtype=np.int32)
labels_train = []
for index in range(len(image_label_list_train)):
    image_label_train = image_label_list_train[index]
    label_train = image_label_train['disease_class']
    num_per_label_train[int(label_train)] += 1
    labels_train.append(int(label_train))

print('trainset:{}'.format(num_per_label_train))
plt.hist(labels_train, bins=classes)
plt.xlabel('label')
plt.ylabel('num_per_label')
plt.title('Plant-Disease-Recognition-Trainset')
plt.show()

# validation
with open(val_label_dir, 'r') as f_val:
    image_label_list_val = json.load(f_val)

num_per_label_val = np.zeros(classes, dtype=np.int32)
labels_val = []
for index in range(len(image_label_list_val)):
    image_label_val = image_label_list_val[index]
    label_val = image_label_val['disease_class']
    num_per_label_val[int(label_val)] += 1
    labels_val.append(int(label_val))

print('validation set:{}'.format(num_per_label_val))
plt.hist(labels_val, bins=classes)
plt.xlabel('label')
plt.ylabel('num_per_label')
plt.title('Plant-Disease-Recognition-Validation Set')
plt.show()

# train set + val set
image_label_list = image_label_list_train + image_label_list_val
num_per_label = np.zeros(classes, dtype=np.int32)
labels = []
for index in range(len(image_label_list)):
    image_label_dict = image_label_list[index]
    label = image_label_dict['disease_class']
    num_per_label[int(label)] += 1
    labels.append(int(label))

print('trainset + validation set:{}'.format(num_per_label))
plt.hist(labels, bins=classes)
plt.xlabel('label')
plt.ylabel('num_per_label')
plt.title('Plant-Disease-Recognition-Trainset and Validation set')
plt.show()
# 
