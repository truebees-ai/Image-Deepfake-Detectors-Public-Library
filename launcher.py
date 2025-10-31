
import os
import subprocess
import glob 
import time
import shutil


def get_stdargs(detector):
    stdargs = {
        'R50_nodown' :  ' --arch res50nodown                            --norm_type resnet --resize_size 256 --resize_ratio 0.75 --resize_prob 0.2 --cmp_qual 30,100  --cmp_prob 0.5 --cropSize 96 --blur_sig 0.1,3.0 --blur_prob 0.5 --jitter_prob 0.8 --colordist_prob 0.2 --cutout_prob 0.2 --noise_prob 0.2',
        'CLIP-D' :      ' --arch opencliplinearnext_clipL14commonpool   --norm_type clip   --resize_size 200 --resize_ratio 1    --resize_prob 0.2 --cmp_qual 65,100  --cmp_prob 0.5 --resizeSize 224', 
        'R50_TF' :      ' --arch nodown                                 --prototype  --freeze',
        'MISLNet' :     '',
        'P2G' : '',
        'NPR' : '',
    }
    return stdargs[detector]

# smi vampire function, busy waiting for a free-enough GPU, use min_vram to set the threshold
def get_gpus():
    from numpy import argwhere, asarray, diff
    import re
    smi = os.popen('nvidia-smi').readlines()
    div = re.compile('[+]-{3,}[+]|[|]={3,}[|]')
    dividers = argwhere([div.match(line) != None for line in smi])[-2:, 0]
    processes = [line for line in smi[dividers[0]+1:dividers[1]] if ' C ' in line]
    free = list(set([process.split()[1] for process in processes]) ^ set([str(0), str(1)]))

    udiv = re.compile('[|]={3,}[+]={3,}[+]={3,}[|]')
    ldiv = re.compile('[+]-{3,}[+]-{3,}[+]-{3,}[+]')
    divider_up = argwhere([udiv.match(line) != None for line in smi])[0,0]
    divider_down = argwhere([ldiv.match(line) != None for line in smi])[-1, 0]

    gpus = [line for line in smi[divider_up+1:divider_down] if '%' in line and 'MiB' in line]
    gpus = [gpu.split('|')[2].replace(' ', '').replace('MiB', '').split('/') for gpu in gpus]
    memory = diff(asarray(gpus).astype(int), axis=1).squeeze()

    return free, memory

def autotest(train_list, data_list):
    assert type(data_list) == list
    task_list = []
    for train in train_list:
        task_list.append({'type':'train', 'details':train})
        for data in data_list:
            name = train['data']
            task_list.append({'type':'test', 'details':{'detector': train['detector'], 'model': name, 'data': data}})

    return task_list

## TB_fixed
# LOCUTUS
# dataset_path = os.path.join(os.sep, 'media', 'mmlab', 'Volume2', 'TrueFake')
# device_override = 'cuda:0'
dataset_path = os.path.join(os.sep, 'media', 'mmlab', 'Datasets_4TB', 'TrueFake')
device_override = 'cuda:1'

# HAL
# dataset_path = os.path.join(os.sep, 'home', 'stefano.dellanna', 'TrueFake')
# device_override = 'cuda:0'

split_file = os.path.abspath('split.json')

os.makedirs('logs', exist_ok=True)

# only list the training/testing to perform
only_list = False
dry_run = False

# # specify the phases to run {train, test}
# phases = ['test']
phases = ['train', 'test']

min_vram = 16000

# device_override = 'cuda:1' # if None, the launcher will automatically select the first available GPU
# device_override = None # if None, the launcher will automatically select the first available GPU

tasks = []

# data = 'gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre'
# data = 'gan2:pre&realFFHQ:pre'
# data = 'sdXL:pre&realFORLAB:pre'

train = [
        # {'detector': 'R50_nodown',      'model': None,     'data': data},
        # {'detector': 'CLIP-D',          'model': None,     'data': data},
        # {'detector': 'MISLNet',         'model': None,     'data': data},
        # {'detector': 'R50_TF',          'model': None,     'data': data},
        # {'detector': 'P2G',             'model': None,     'data': data},
        # {'detector': 'NPR',             'model': None,     'data': data},
        ]

train = [
        # {'detector': 'R50_TF',          'model': None,     'data': 'sdXL:pre&realFORLAB:pre'},
        # {'detector': 'R50_TF',          'model': None,     'data': 'gan2:pre&realFFHQ:pre'},
        {'detector': 'R50_TF',          'model': None,     'data': 'gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre'},
        # {'detector': 'R50_nodown',          'model': None,     'data': 'gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre'},
        ]

test_list = [
            'realFFHQ:fb',  'realFORLAB:fb',  'gan1:fb',  'gan2:fb',  'gan3:fb',  'sd15:fb',  'sd2:fb',   'sd3:fb',   'sdXL:fb',  'flux:fb',
            'realFFHQ:tl',  'realFORLAB:tl',  'gan1:tl',  'gan2:tl',  'gan3:tl',  'sd15:tl',  'sd2:tl',   'sd3:tl',   'sdXL:tl',  'flux:tl',
            'realFFHQ:tw',  'realFORLAB:tw',  'gan1:tw',  'gan2:tw',  'gan3:tw',  'sd15:tw',  'sd2:tw',   'sd3:tw',   'sdXL:tw',  'flux:tw',
            'realFFHQ:pre', 'realFORLAB:pre', 'gan1:pre', 'gan2:pre', 'gan3:pre', 'sd15:pre', 'sd2:pre',  'sd3:pre',  'sdXL:pre', 'flux:pre',
            ]

# test_list = ['realFORLAB:fb', 'realFORLAB:tl', 'realFORLAB:tw']

tasks.extend(autotest(train, test_list))


# # phase filter
# tasks = [task for task in tasks if task['type'] in phases]

print('Number of tasks:', len(tasks))
for task in tasks:
    print(task)

if only_list:
    quit()

# from here the launcher will create all the arguments to use when calling the train script
for task in tasks:

    if task['type'] not in phases:
        continue
    args = []
    if task['type'] == 'train':
        args.append(f'--name "{task["details"]["data"]}_freeze_down"')
        # args.append(f'--name "{task["details"]["data"]}"')
    else:
        args.append(f'--name "{task["details"]["model"]}_freeze_down"')
        # args.append(f'--name "{task["details"]["model"]}"')

    args.append(f'--split_file {split_file}')
    args.append(f'--task {task["type"]}')
    args.append(f'--num_threads {8}')
    args.append(f'--data_keys "{task["details"]["data"]}"')
    args.append(f'--data_root {dataset_path}')
    # args.append(f'--num_epoches {5}')

    device = None
    if not device_override is None:
        device = device_override
    else:
        if not dry_run:
            print('Waiting for GPU')
            while(device == None):
                free, memory = get_gpus()
                if len(free):
                    device = "cuda:"+free[0]
                elif max(memory) > min_vram:
                    device = "cuda:" + str([i for i, mem in enumerate(memory) if mem == max(memory)][0])
                time.sleep(1)
            print('GPU found')
    
    args.append(f'--device {device}')
    args = ' '.join(args + get_stdargs(task["details"]["detector"]).split()) 

    # call to train.py
    if not dry_run:
        log_file = f'logs/{task["type"]}_{task["details"]["detector"]}_{task["details"]["model"]}_{task["details"]["data"]}.log'
        with open(log_file, 'w') as f:
            cwd = os.getcwd()
            os.chdir(f'./detectors/{task["details"]["detector"]}')

            start_time = time.time()

            runner = f'{task["type"]}.py'   
            print(f'Call to {runner} with: {args}')
            # subprocess.run(f'python -u {runner} {args}', shell=True)
            subprocess.run(f'python -u {runner} {args}', shell=True, stdout=f, stderr=f)

            end_time = time.time()
            print(f'Execution time: {end_time-start_time:.2f} seconds')

            print('#'*80)
            print('#'*80)
            # subprocess.run(f'python -u {task["type"]}.py {args}', shell=True)

            os.chdir(cwd)



