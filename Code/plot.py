# -*- coding:utf-8 -*-

import json
from matplotlib import pyplot as plt
import numpy as np

label_dir = '/media/gfx/GFX/DATASET/Plant_Disease_Recognition/ai_challenger_pdr2018_train_annotations_20181021.json'
save_dir = './'

classes = 61

with open(label_dir, 'r') as f:
    image_label_list = json.load(f)

num_per_label = np.zeros(classes, dtype=np.int32)
labels = []
for index in range(len(image_label_list)):
    image_label_dict = image_label_list[index]
    label = image_label_dict['disease_class']
    num_per_label[int(label)] += 1
    labels.append(int(label))

print(num_per_label)
plt.hist(labels, bins=classes)
plt.xlabel('label')
plt.ylabel('num_per_label')
plt.title('Plant-Disease-Recognition')
plt.show()


