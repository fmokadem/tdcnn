import os
import re
import json
import torch
import torch.nn as nn
from datetime import datetime
from torchvision import models
from common.utils import get_conv2d_layers

def create_model(model_name):
    """Create a model without loading weights"""
    if model_name == 'vgg16' or model_name == 'vgg':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, 10)
    elif model_name == 'alexnet':
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(4096, 10)
    elif model_name == 'resnet18' or model_name == 'resnet' or model_name == 'renet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 10)
    else:
        raise ValueError(f"Model '{model_name}' not supported")
    return model

def get_model_conv_layers(model_name):
    """Get Conv2d layers from a model and their cin/cout dimensions"""
    # Create model without weights
    model = create_model(model_name)
    conv_layers = get_conv2d_layers(model)
    
    layer_dims = {}
    for name, layer in conv_layers.items():
        layer_dims[name] = {
            'cin': layer.in_channels,
            'cout': layer.out_channels
        }
    return layer_dims

def compute_compression_rate(layer_dims, config):
    """Compute compression rate as (cin * cout)/(rin * rout) for each layer"""
    total_uncompressed = 0
    total_compressed = 0
    
    for layer_name, rank in config.items():
        cin = layer_dims[layer_name]['cin']
        cout = layer_dims[layer_name]['cout']
        total_uncompressed += cin * cout
        total_compressed += rank[0] * rank[1]  # rin, rout from config tuple
    
    return total_uncompressed / total_compressed if total_compressed > 0 else 0

def parse_config_str(config_str):
    """Parse a config string into a dictionary of layer names and rank tuples"""
    config = {}
    
    # Define patterns for different model architectures
    patterns = {
        'vgg': r'(features\.\d+):\((\d+),\s*(\d+)\)',
        'alexnet': r'(features\.\d+):\((\d+),\s*(\d+)\)',
        'resnet': r'(layer\d+\.\d+\.conv\d+):\((\d+),\s*(\d+)\)'
    }
    
    # Try each pattern until we find matches
    for pattern in patterns.values():
        matches = re.finditer(pattern, config_str)
        match_count = 0
        for match in matches:
            layer_name = match.group(1)
            rin = int(match.group(2))
            rout = int(match.group(3))
            config[layer_name] = (rin, rout)
            match_count += 1
        
        if match_count > 0:
            break
    
    if not config:
        print(f"Warning: No layer configurations found in string: {config_str}")
    
    return config

def parse_log_file(log_file):
    results = []
    with open(log_file, 'r') as f:
        content = f.read()
        # Find all compressed_model JSON blocks
        pattern = r'compressed_model:\s*\{([^}]+)\}'
        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            json_str = '{' + match.group(1) + '}'
            # Remove newlines and extra whitespace to normalize the JSON
            json_str = re.sub(r'\s+', ' ', json_str)
            try:
                result = json.loads(json_str)
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON: {e}")
                continue
    
    return results

def main():
    log_dir = os.path.join(os.path.dirname(__file__), 'results/exp_20250330_115549')
    output_dir = log_dir
    
    # Process each model type
    for model_name in ['vgg16', 'resnet18', 'alexnet']:
        print(f"\nProcessing {model_name}...")
        
        # Get layer dimensions for the model
        layer_dims = get_model_conv_layers(model_name)
        print(f"Found layer dimensions for {model_name}")
        
        # Find corresponding log file
        log_files = [f for f in os.listdir(log_dir) if f.startswith(f'htd_{model_name.replace("resnet18", "renet18")}_') and f.endswith('.log')]
        
        if not log_files:
            print(f"No log files found for {model_name}")
            continue
            
        # Process the most recent log file
        log_file = sorted(log_files)[-1]
        print(f"Processing log file: {log_file}")
        
        try:
            # Parse results from log file
            results = parse_log_file(os.path.join(log_dir, log_file))
            print(f"Parsed {len(results)} results from log file")
            
            # Compute accurate compression rates for each result
            processed_results = []
            for result in results:
                config = parse_config_str(result['config_str'])
                compression_rate = compute_compression_rate(layer_dims, config)
                
                processed_results.append({
                    'config': config,
                    'accuracy': result['accuracy'],
                    'inference_time': result['inference_time'],
                    'params': result['params'],
                    'flops': result['flops'],
                    'compression_rate': compression_rate
                })
            
            # Save results to JSON file
            output_file = os.path.join(output_dir, f'htd-{model_name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            with open(output_file, 'w') as f:
                json.dump({
                    'model': model_name,
                    'layer_dimensions': layer_dims,
                    'results': processed_results
                }, f, indent=2)
            print(f"Saved results to {output_file}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue

if __name__ == '__main__':
    main() 