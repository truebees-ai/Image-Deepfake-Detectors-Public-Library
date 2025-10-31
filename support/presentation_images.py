import os
import random
import cv2
import numpy as np
import subprocess
import shutil
from skimage.metrics import structural_similarity 
root_pre = '/media/mmlab/Datasets_4TB/TrueFake/PreSocial'

images = {}
for dataset_root, dataset_dirs, dataset_files in os.walk(root_pre, topdown=True, followlinks=True):
    if len(dataset_dirs):
        continue

    (label, gen, sub)  = f'{dataset_root}/'.replace(root_pre + os.sep, '').split(os.sep)[:3]
    
    if f'{gen}' not in images:
        images[f'{gen}'] = []
    images[f'{gen}'].extend([os.path.join(label, gen, sub, file) for file in dataset_files])

to_select = 20

to_select = [random.randint(0, 30719) for _ in range(4)]
# to_select = list(range(1,5))
selected_images = {}
for gen_key, gen_images in images.items():
    print(f'{gen_key}: {len(gen_images)}')
    #random.shuffle(gen_images)
    selected_images[gen_key] = [gen_images[idx] for idx in to_select]
    # print(gen_images[:to_select])

for gen_key, gen_images in selected_images.items():
    for idx, image_name in enumerate(gen_images):
        image_path = os.path.join(root_pre, image_name)
        print(image_path)
        shutil.copy(image_path, f'./images/{gen_key}_{idx}.png')