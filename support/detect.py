import os
import json
import sys
import yaml
import subprocess
import shlex
from datetime import datetime

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_detect(args):
    """Run single image detection."""
    if not args.image:
        raise ValueError("--image is required for detect mode")
    
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
        
    # Load detector config
    config_path = os.path.join(args.config_dir, f'{args.detector}.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = load_config(config_path)
    global_config = config.get('global', {})
    
    # Set device
    device = args.device or global_config.get('device_override') or 'auto'
    if device in ('null', '', 'auto') and not args.dry_run:
        import torch
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
    # Set up output path for results
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract image filename without extension
        image_name = os.path.splitext(os.path.basename(args.image))[0]
        output_dir = os.path.join('detection_results', args.detector, 'detect')
        os.makedirs(output_dir, exist_ok=True)
        args.output = os.path.join(output_dir, f'{timestamp}_{image_name}.json')
    
    # Call detector's detect.py
    detector_path = os.path.join('detectors', args.detector)
    detect_script = os.path.join(detector_path, 'detect.py')
    
    if not os.path.exists(detect_script):
        raise FileNotFoundError(f"Detector {args.detector} does not support single image detection")
    
    cmd_args = [sys.executable, detect_script, '--image', os.path.abspath(args.image),
                '--device', device, '--output', os.path.abspath(args.output)]
    
    # Add model path if specified
    if args.weights:
        cmd_args.extend(['--model', args.weights])
    if args.detector in ('PGC', 'ForensicConcept', 'MDMF') and args.batch_size:
        cmd_args.extend(['--batch_size', str(args.batch_size)])
    
    cmd = shlex.join(cmd_args)
    print(f"Running detection with {args.detector}...")
    
    if args.dry_run:
        print(cmd)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        subprocess.run(cmd_args, check=True)
        if not os.path.isfile(args.output):
            raise RuntimeError(f"Detector completed without writing a result: {args.output}")
        
        # Print results if available
        if os.path.exists(args.output):
            with open(args.output, 'r') as f:
                result = json.load(f)
            print("\nDetection Results:")
            print(f"Prediction: {result['prediction']}")
            if 'fake_score' not in result:
                result['fake_score'] = result['confidence']
                with open(args.output, 'w') as handle:
                    json.dump(result, handle, indent=2)
            print(f"Fake score: {result['fake_score']:.4f}")
            print(f"Time: {result['elapsed_time']:.3f}s")
            print(f"\nFull results saved to: {args.output}")
