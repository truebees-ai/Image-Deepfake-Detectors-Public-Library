
import os
import subprocess
import time
import argparse
import yaml
import glob
import shutil
import sys
import shlex
from pathlib import Path

METHODS = ['R50_TF', 'R50_nodown', 'CLIP-D', 'P2G', 'NPR', 'PGC', 'ForensicConcept', 'MDMF']


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def autotest(train_list, data_list, detector_name, checkpoint_name):
    """Generate task list from training and testing configurations."""
    assert type(data_list) == list
    task_list = []

    for train_config in train_list:
        
        train_dict = {'detector': detector_name, 'model': None, 'data': train_config['data']}
        task_list.append({'type':'train', 'details':train_dict})
        
        for data in data_list:
            
            name = checkpoint_name #train_dict['data']
            task_list.append({'type':'test', 'details':{'detector': detector_name, 'model': name, 'data': data}})
    
    return task_list


def parse_phases(phases_str):
    """Parse phases string into list."""
    if phases_str.lower() == 'both':
        return ['train', 'test']
    elif phases_str.lower() == 'train':
        return ['train']
    elif phases_str.lower() == 'test':
        return ['test']
    else:
        raise ValueError(f"Invalid phases: {phases_str}. Must be 'train', 'test', or 'both'")


def run_demo(args):
    import json
    import torch

    project_root = os.path.abspath(os.path.dirname(__file__))
    demo_root = os.path.join(project_root, 'demo_images')
    assert os.path.isdir(demo_root), f"Demo folder not found: {demo_root}"

    # Build split file from demo_images
    def build_demo_split_json(root_path, out_path):
        test_entries = []
        for mod in ['PreSocial', 'Facebook', 'Telegram', 'X']:
            mod_path = os.path.join(root_path, mod)
            if not os.path.isdir(mod_path):
                continue
            for dirpath, dirnames, filenames in os.walk(mod_path, topdown=True, followlinks=True):
                if len(dirnames):
                    continue
                rel_dir = f"{dirpath}/".replace(mod_path + os.sep, '')
                parts = rel_dir.split(os.sep)[:3]
                if len(parts) < 3:
                    continue
                label, gen, sub = parts
                for fname in sorted(filenames):
                    ext = os.path.splitext(fname)[1].lower()
                    if ext not in ['.png', '.jpg', '.jpeg']:
                        continue
                    stem = os.path.splitext(fname)[0]
                    test_entries.append(os.path.join(gen, sub, stem))

        with open(out_path, 'w') as f:
            json.dump({'test': sorted(list(set(test_entries)))}, f)

    split_demo_file = os.path.join(project_root, 'split_demo.json')
    build_demo_split_json(demo_root, split_demo_file)

    def prepare_best_checkpoint(detector_dir, preferred_path=None):
        weights_dir = os.path.join(detector_dir, 'checkpoint', 'pretrained', 'weights')
        src_weight = None

        if preferred_path:
            src_weight = preferred_path if os.path.isabs(preferred_path) else os.path.normpath(os.path.join(detector_dir, preferred_path))
            if not os.path.isfile(src_weight):
                print(f"[demo] Preferred weights not found at {src_weight}, falling back to search")
                src_weight = None

        if src_weight is None:
            if not os.path.isdir(weights_dir):
                return None
            candidates = []
            for ext in ('*.pt', '*.pth'):
                candidates.extend(glob.glob(os.path.join(weights_dir, ext)))
            if not candidates:
                return None
            src_weight = sorted(candidates)[0]

        run_dir = os.path.join(detector_dir, 'checkpoint', 'demo', 'weights')
        os.makedirs(run_dir, exist_ok=True)
        dst_weight = os.path.join(run_dir, 'best.pt')
        shutil.copy2(src_weight, dst_weight)
        return dst_weight

    
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    name = 'demo'

    detectors_root = os.path.join(project_root, 'detectors')
    all_methods = METHODS
    methods = all_methods if args.demo_detector == 'all' else [args.demo_detector]

    os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)

    for method in methods:
        det_dir = os.path.join(detectors_root, method)
        if not os.path.isdir(det_dir):
            continue

        if method == 'P2G' and not str(device).startswith('cuda'):
            message = f"[demo] P2G requires CUDA (device is {device})"
            if args.demo_detector != 'all':
                raise RuntimeError(message)
            print(f"{message}; skipping")
            continue

        recent = method in ('PGC', 'ForensicConcept', 'MDMF')
        run_name = args.weights_name or 'pretrained'
        preferred_weights = args.weights_name or './checkpoint/pretrained/weights/best.pt'
        best_path = (os.path.join(det_dir, 'checkpoint', run_name, 'weights', 'best.pt')
                     if recent else prepare_best_checkpoint(det_dir, preferred_weights))
        if best_path is None or not os.path.isfile(best_path):
            message = f"[demo] No weights for {method}: {best_path or 'checkpoint/pretrained/weights/best.pt'}"
            if args.demo_detector != 'all':
                raise FileNotFoundError(message)
            print(f"{message}; skipping")
            continue

        config_path = os.path.join(args.config_dir, f'{method}.yaml')
        config = load_config(config_path) if os.path.exists(config_path) else {}
        detector_args = config.get('detector_args', [])
        testing_keys = ['all:all']
        global_cfg = config.get('global', {})
        num_threads = global_cfg.get('num_threads', 8)

        for data_keys in testing_keys:
            args_list = ['--name', run_name if recent else name, '--task', 'test',
                         '--device', device, '--split_file', split_demo_file,
                         '--data_root', demo_root, '--data_keys', data_keys,
                         '--num_threads', str(num_threads), *detector_args]
            if recent and args.batch_size:
                args_list.extend(['--batch_size', str(args.batch_size)])
            elif method in ('R50_TF', 'R50_nodown', 'CLIP-D', 'NPR'):
                # Demo images have mixed sizes and these test loaders do not crop, so they cannot be batched.
                args_list.extend(['--batch_size', '1'])
            log_file = os.path.join(project_root, 'logs', f'demo_{method}_{data_keys.replace(":","-")}.log')
            with open(log_file, 'w') as f:
                print(f"[demo] Running {method} test with args: {shlex.join(args_list)}")
                subprocess.run([sys.executable, '-u', 'test.py', *args_list], cwd=det_dir, check=True)
        if not recent:
            shutil.rmtree(os.path.join(det_dir, 'checkpoint', 'demo'))

    print('[demo] Completed. Results saved under detectors/<method>/results/<run>/all-all/results.csv')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Launcher for deepfake detector training and testing')
    parser.add_argument('--detector', type=str, required=False,
                        choices=METHODS,
                        help='Detector to use')
    parser.add_argument('--phases', type=str, default='both',
                        choices=['train', 'test', 'both'],
                        help='Phases to run: train, test, or both (default: both)')
    parser.add_argument('--config-dir', type=str, default='configs',
                        help='Path to configs directory (default: configs/)'),
    parser.add_argument('--weights_name', type=str, default=None, 
                        help='Name of the weights directory')
    parser.add_argument('--demo', action='store_true', help='Run demo on demo_images across detectors')
    parser.add_argument('--demo-detector', type=str, default='all', choices=['all', *METHODS], help='Which detector to demo (default: all)')
    parser.add_argument('--device', default=None, help='auto, cpu, cuda:N or mps; overrides YAML device_override')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for recent detectors')
    
    # Add detect mode arguments
    detect_group = parser.add_argument_group('detect', 'Single image detection options')
    detect_group.add_argument('--detect', action='store_true', help='Run single image detection mode')
    detect_group.add_argument('--image', type=str, help='Path to image file for detection')
    detect_group.add_argument('--weights', type=str, default='pretrained', help='Path to model weights for detection')
    detect_group.add_argument('--output', type=str, help='Path to save detection results')
    detect_group.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    
    args = parser.parse_args()

    if args.demo:
        return run_demo(args)
        
    if args.detect:
        if args.detector is None:
            parser.error('--detector is required for detect mode')
        if args.image is None:
            parser.error('--image is required for detect mode')
        from support.detect import run_detect
        return run_detect(args)

    if args.detector is None:
        parser.error('--detector is required unless --demo is specified')

    # Load configuration from YAML
    config_path = os.path.join(args.config_dir, f'{args.detector}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    
    # Extract configuration values
    global_config = config.get('global', {})
    dataset_path = global_config.get('dataset_path')
    device_override = args.device or global_config.get('device_override') or 'auto'
    if args.weights_name is not None:
        global_config['name'] = args.weights_name
    else:
        global_config['name'] = config.get('training', [])[0]['data']
    model_name = global_config.get('name')
    if device_override in ('null', ''):
        device_override = 'auto'
    split_file = os.path.abspath(global_config.get('split_file', 'split.json'))
    num_threads = global_config.get('num_threads', 8)
    dry_run = args.dry_run or global_config.get('dry_run', False)
    only_list = global_config.get('only_list', False)
    phases = parse_phases(args.phases)
    
    detector_args = config.get('detector_args', [])
    training_configs = config.get('training', [])
    test_list = config.get('testing', [])
    
    if not dry_run:
        os.makedirs('logs', exist_ok=True)
    
    # Generate tasks
    tasks = []
    if training_configs:
        tasks.extend(autotest(training_configs, test_list, args.detector, model_name))
    
    print('Number of tasks:', len(tasks))
    for task in tasks:
        print(task)
    
    if only_list:
        return
    
    # From here the launcher will create all the arguments to use when calling the train script
    for task in tasks:
        if task['type'] not in phases:
            continue
        
        cmd_args = []
        
        cmd_args.extend(['--name', str(model_name if task['type'] == 'train' else task['details']['model'])])
        
        cmd_args.extend(['--split_file', split_file, '--task', task['type'],
                         '--num_threads', str(num_threads), '--data_keys', task['details']['data'],
                         '--data_root', str(dataset_path)])
        
        device = device_override
        if device == 'auto' and not dry_run:
            import torch
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        cmd_args.extend(['--device', device])
        if args.detector in ('PGC', 'ForensicConcept', 'MDMF') and args.batch_size:
            cmd_args.extend(['--batch_size', str(args.batch_size)])
        
        # Add detector-specific arguments
        cmd_args.extend(detector_args)
        
        cmd_args_str = shlex.join(cmd_args)
        
        # Call train.py or test.py
        if dry_run:
            print(shlex.join([sys.executable, '-u', f'{task["type"]}.py', *cmd_args]))
        else:
            if not Path(dataset_path).is_dir():
                raise FileNotFoundError(f'Dataset directory not found: {dataset_path}')
            if not Path(split_file).is_file():
                raise FileNotFoundError(f'Split file not found: {split_file}')
            #log_file = f'logs/{task["type"]}_{task["details"]["detector"]}_{task["details"]["model"]}_{task["details"]["data"]}.log'
            log_file = f'logs/{task["type"]}_{task["details"]["detector"]}_{model_name}_{task["details"]["data"]}.log'
            with open(log_file, 'w') as f:
                start_time = time.time()
                
                runner = f'{task["type"]}.py'
                print(f'Call to {runner} with: {cmd_args_str}')
                
                subprocess.run([sys.executable, '-u', runner, *cmd_args],
                               cwd=os.path.join('detectors', task['details']['detector']), check=True)
                
                end_time = time.time()
                print(f'Execution time: {end_time-start_time:.2f} seconds')
                
                print('#'*80)
                print('#'*80)
                


if __name__ == '__main__':
    main()



