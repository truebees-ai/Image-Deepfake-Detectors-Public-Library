import os
import random
import cv2
import numpy as np
import subprocess
from skimage.metrics import structural_similarity 
root_pre = '/media/mmlab/Datasets_4TB/TrueFake/PreSocial'

root_fb = root_pre.replace('PreSocial', 'Facebook')
root_tl = root_pre.replace('PreSocial', 'Telegram')
root_tw = root_pre.replace('PreSocial', 'Twitter')

images = {}
for dataset_root, dataset_dirs, dataset_files in os.walk(root_tw, topdown=True, followlinks=True):
    if len(dataset_dirs):
        continue

    (label, gen, sub)  = f'{dataset_root}/'.replace(root_tw + os.sep, '').split(os.sep)[:3]
    if gen not in ['StableDiffusion3']:
        continue
    
    if f'{gen}' not in images:
        images[f'{gen}'] = []
    images[f'{gen}'].extend([os.path.join(label, gen, sub, file) for file in dataset_files])

to_select = 200
selected_images = {}
for gen_key, gen_images in images.items():
    print(f'{gen_key}: {len(gen_images)}')
    random.shuffle(gen_images)
    selected_images[gen_key] = gen_images[:to_select]
    # print(gen_images[:to_select])

compression = {}

# for social_key, social_root in zip(['fb', 'tl', 'tw'], [root_fb, root_tl, root_tw]):
for social_key, social_root in zip(['fb'], [root_fb]):
    print(social_key)
    for gen_key, gen_images in selected_images.items():
        print(gen_key)
        print(gen_images)
        for idx, image in enumerate(gen_images):
            # print(idx)
            shared_image = os.path.join(social_root, image)
            original_image = os.path.join(root_pre, image).replace('.jpg', '.png' if 'FORLAB' not in image else '.jpg') 
            assert os.path.exists(original_image), f'Image not found: {original_image}'
            assert os.path.exists(shared_image), f'Image not found: {shared_image}'

            cmd = f'/home/stefano.dellanna/bin/magick identify -verbose {shared_image} | grep Quality'
            # print(os.system(cmd))

            subprocess.run(cmd, shell=True)

            # shared_image = cv2.imread(shared_image)
            # original_image = cv2.imread(original_image)

            # if shared_image.shape != original_image.shape:
            #     print('Resizing applied:', original_image.shape, shared_image.shape)
#                 shared_image = cv2.resize(shared_image, (original_image.shape[1], original_image.shape[0]))

# #             # compute PSNR, MSE and SSIM
# #             mse = np.mean((shared_image - original_image) ** 2)
# #             if mse == 0:
# #                 psnr = 100
# #             else:
# #                 psnr = 20 * np.log10(255.0 / np.sqrt(mse))
# #             (ssim, _) = structural_similarity(original_image, shared_image, channel_axis=2, full=True)

# #             # print(f'{gen_key} {image} PSNR: {psnr:.2f} SSIM: {ssim:.2f} MSE: {mse:.2f}')

# #             if gen_key not in compression:
# #                 compression[f'{social_key}_{gen_key}'] = {'psnr': [], 'ssim': [], 'mse': []}
            
# #             compression[f'{social_key}_{gen_key}']['psnr'].append(psnr)
# #             compression[f'{social_key}_{gen_key}']['ssim'].append(ssim)
# #             compression[f'{social_key}_{gen_key}']['mse'].append(mse)

# # for data_key, gen_compression in compression.items():
# #     print(f'{data_key} PSNR: {np.mean(gen_compression["psnr"]):.2f} SSIM: {np.mean(gen_compression["ssim"]):.2f} MSE: {np.mean(gen_compression["mse"]):.2f}')