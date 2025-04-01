#!/usr/bin/env python
# coding: utf-8

# ## Experiment
# - htd on alexnet 

import sys

EXP_DIR = "/home/fmokadem/NAS/tdcnn/"
sys.path.append(EXP_DIR)

import os
import sys
import torch
from torchvision import models
import time
import random
import numpy as np
import itertools
import math
import argparse
from datetime import datetime
import json
import copy

from common.dataset import load_mnist
from common._logging import setup_logger
from common.utils import (
    count_parameters, 
    measure_inference_time, 
    calculate_accuracy, 
    get_flops, 
    get_conv2d_layers,
    infer_rank, 
    calculate_layer_params,
    replace_conv2d_with_tucker,
    fine_tune
)
from common.load_models import load_model

MODEL_NAME = 'alexnet'
FINETUNE = True
MAX_CFG = 250
ACCU_RQT = .90

MODEL_PATH = os.path.join(EXP_DIR, f'finetuned/saved_models/{MODEL_NAME}_mnist.pth')
LOG_DIR = os.path.join(EXP_DIR, 'logs')
LOG_PREFIX = 'htd_alexnet'

logger = setup_logger(LOG_PREFIX, LOG_DIR, LOG_PREFIX)
logger.info(f"Starting HTD experiment for {MODEL_NAME}")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# use gpu 0 only
device_idx = 0

if torch.cuda.is_available():
    torch.cuda.set_device(device_idx)
    device = f'cuda:{device_idx}'
else:
    device = "cpu"

logger.info(f"Using device: {device}")

# Load dataset
train_loader, test_loader = load_mnist()
logger.info(f"MNIST loaded: {len(train_loader.dataset)} train, {len(test_loader.dataset)} test samples")

# Load model
model = load_model(MODEL_NAME, MODEL_PATH, device)
logger.info(f"Loaded {MODEL_NAME} from {MODEL_PATH}")

train_loader.dataset

from torchsummary import summary
summary(model, (3, 244, 244))

# Extract Conv2D layers
conv_layers = get_conv2d_layers(model)
logger.info(f"Found {len(conv_layers)} Conv2D layers in {MODEL_NAME}")
conv_layers

# Initialize layer information
# Conv2 layers in Pytorch are (Cin, Cout, ks, ks), i.e. a 4D tensor with rank 4, called modes 0 to 3
# we are interested in low rank approximating of modes 0 and 1, i.e. compressing the information in  the channels
# each mode is almost always full rank, i.e. of rank == size
# therefore for the pupose of this exp rank of a layer is the min(Cin, Cout) 

# TODO: complexity ranks in decreasing order layers that are closest to the middle.  
layer_info = {}
for name, layer in conv_layers.items():
    r_i = infer_rank(layer)
    layer_info[name] = {
        'layer': layer,
        'r_i': r_i,
        'params': calculate_layer_params(layer),
        'complexity': None
    }
    logger.info(f"Layer {name}: initial rank R_i = {r_i}, parameters = {layer_info[name]['params']}")

layer_info

# Compute baseline metrics
baseline_params = count_parameters(model)
baseline_flops = get_flops(model)
baseline_accuracy = calculate_accuracy(model, test_loader, device)
baseline_inference_time = measure_inference_time(model, test_loader, device, num_runs=3)

logger.info(f"Baseline {MODEL_NAME}: params={baseline_params}, "
            f"FLOPs={baseline_flops}, accuracy={baseline_accuracy:.4f}, "
            f"inference_time={baseline_inference_time:.4f}s")

# Timestamp for unique file naming
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Generate possible ranks per layer, that is 1 up to it's rank - 1 
# if layer is of rank 1, then possible ranks are just 1
possible_ranks = {}
for name, info in layer_info.items():
    r_i = info['r_i']
    ranks = [1] + list(range(2, r_i)) 
    possible_ranks[name] = ranks

total_possible_configs = np.prod([max(1, layer_info[name]['r_i'] - 1) for name in layer_info.keys()])
num_configs_to_try = min(total_possible_configs, MAX_CFG)
logger.info(f"Total possible configurations: {total_possible_configs}, will try: {num_configs_to_try}")

from torch import nn
def construct_layer_dict(model):
    layer_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            W = module.weight.data
            cin = module.in_channels
            cout = module.out_channels
            layer_dict[name] = (W, cin, cout)
    return layer_dict

layer_dict = construct_layer_dict(model)

from tensorly.decomposition import partial_tucker
import tensorly as tl    
from tensorly.tucker_tensor import tucker_to_tensor
from tensorly.metrics.regression import MSE

tl.set_backend('pytorch')

# Heuristic: layers with bigger cin*cout more suseptible to lower ranks 
def get_size_based_probs(cin, cout, rank_candidates, beta=1.0):
    size = cin * cout
    preference = np.log(size + 1)  # +1 to avoid log(0), though unlikely
    scores = [-beta * r / preference for r in rank_candidates]
    probs = np.exp(scores) / np.sum(np.exp(scores))
    return probs

# Heuristic: Layers with higher reconstruction erros are less suseptible to low ranks 
def compute_sensitivity(W, cin, cout):
    rank = [max(1, cout // 2), max(1, cin // 2)]
    (core, factors) = partial_tucker(W, modes=[0, 1], rank=rank, init='svd')
       
    reconstructed_W = tucker_to_tensor(core, factors)
    return MSE(W, reconstructed_W) 

# Precompute sensitivity for all layers
def precompute_sensitivities(layer_dict):
    sensitivities = {}
    for name, (W, cin, cout) in layer_dict.items():
        sensitivities[name] = compute_sensitivity(W, cin, cout).to('cpu')
    # Normalize sensitivities to [0, 1]
    max_error = max(list(sensitivities.values()))
    if max_error > 0:  # Avoid division by zero
        sensitivities = {name: err / max_error for name, err in sensitivities.items()}
    return sensitivities
    
def get_sensitivity_probs(sensitivity, rank_candidates, alpha=1.0):
    max_rank = max(rank_candidates)
    scores = [(r / max_rank) ** (alpha * sensitivity) for r in rank_candidates]
    probs = scores / np.sum(scores)
    return probs

def get_rank_candidates(channels, r=(75, 45, -5)):
    s, f, stride = r
    percentages = np.arange(s, f, stride) / 100
    candidates = set([max(1, int(channels * p)) for p in percentages])
    return candidates

# Generator function to yield num_cfg configurations
def generate_configs(layer_dict, num_cfg):
    # Precompute rank candidates for each layer
    rank_candidates = {}
    for name, (W, cin, cout) in layer_dict.items():
        cin_candidates = get_rank_candidates(cin)  
        cout_candidates = get_rank_candidates(cout)  
        
        rank_pairs = list(itertools.product(cin_candidates, cout_candidates))
        # we sort to keep higher ranks on top of the search 
        # no heuristic, budget num_cfg accross all layers equally 
        # num_cfg / len(layer_dict.keys) searches per layer
        spl = int(num_cfg ** (1 / len(layer_dict.keys())))
        rank_candidates[name] = sorted(rank_pairs, key=lambda x: (-x[0], -x[1]))[:spl]
    
    # Yield exactly num_cfg random configurations
    for config in itertools.product(*rank_candidates.values()):
        yield dict(zip(layer_dict.keys(), config))

def duplicate_model(model):
    # Check if the model name is valid
    model_name = MODEL_NAME
    
    if model_name == 'vgg':
        model_cp = models.vgg16(weights=None)
        model_cp.classifier[6] = nn.Linear(4096, 10)
        
    elif model_name == 'alexnet':
        model_cp = models.alexnet(weights=None)
        model_cp.classifier[6] = nn.Linear(4096, 10)
        
    elif model_name == 'resnet':
        model_cp = models.resnet18(weights=None)
        model_cp.fc = nn.Linear(512, 10)

    model_cp.load_state_dict(model.state_dict())   
    return model_cp

configs = generate_configs(layer_dict, MAX_CFG)
next(configs)

def process_config(config, device_idx):
    try:        
        # Explicitly create a new scope to help with garbage collection
        with torch.enable_grad():

            config_str = ", ".join([f"{k}:{v}" for k, v in config.items()])
            logger.info(f"Compressing to:{config_str}")
            
            # Apply the configuration
            model.to('cpu')
            compressed_model = duplicate_model(model)
            compressed_model.to(device)

            for name, rank in config.items():
                layer = layer_info[name]['layer']
                compressed_model = replace_conv2d_with_tucker(compressed_model, name, layer, rank)

            # verify compressed model is still on gpu
            compressed_model.to(device)

            # Finetune for 3 epochs 
            if FINETUNE: 
                logger.info(f"finetuning:{config_str}")
                compressed_model = fine_tune(compressed_model, train_loader, device, epochs=3, lr=0.001)
                
            # Evaluate the model
            accuracy = calculate_accuracy(compressed_model, test_loader, device)
            params = count_parameters(compressed_model)
            flops = get_flops(compressed_model)
            inference_time = measure_inference_time(compressed_model, test_loader, device, num_runs=3)
            compression_rate = baseline_params / params if params > 0 else float('inf')
            
            result = {
                'config_str': config_str,
                'params': params,
                'flops': flops,
                'accuracy': accuracy,
                'inference_time': inference_time,
                'compression_rate': compression_rate,
                'accepted': True if accuracy >= acceptance_threshold else False
            }
            result_str = json.dumps(result, indent=4, default=str)
            logger.info(f"compressed_model:\n{result_str}")

            return result
    
    except Exception as e:
        logger.error(f"Error processing config: {config}. Error: {str(e)}")
        return None
    
    finally:
        # Explicit cleanup
        if 'compressed_model' in locals():
            del compressed_model
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

num_workers = 4
num_gpus = 1  # torch.cuda.device_count() if torch.cuda.is_available() else 0
results = []
tried_count = 0
accepted_models = []

configs = generate_configs(layer_dict, MAX_CFG)
acceptance_threshold = ACCU_RQT * baseline_accuracy

with ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Submit tasks with device indices in round-robin
    if num_gpus > 0:
        futures = [executor.submit(process_config, config, 0) for config in configs] #- i % num_gpus) for i, config in enumerate(configs)]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    
                    if result['accepted']:
                        accepted_models.append(result)
                    
                    tried_count += 1
                    if tried_count % 10 == 0:
                        logger.info(f"Evaluated {tried_count} configurations, found {len(accepted_models)} accepted models")
            
            except Exception as e:
                logger.error(f"Error in future processing: {str(e)}")
    else:
        logger.warning('NO GPU')

# Save top 10 models
if accepted_models:
    accepted_models.sort(key=lambda x: x['score'], reverse=True)
    top_models = accepted_models[:10]
    logger.info(f"Found {len(accepted_models)} accepted models, saving top {len(top_models)}")
    
    for model_info in top_models:
        final_save_path = os.path.join(save_dir, f"htd_{MODEL_NAME}_{model_info['config_str']}_{timestamp}.pth")
        os.rename(model_info['model_path'], final_save_path)
        logger.info(f"Saved top model {model_info['config_str']} to {final_save_path}")/home/fmokadem/NAS/tdcnn/README.md
    
    # Clean up remaining temporary files
    for model_info in accepted_models[10:]:
        if model_info['model_path'] and os.path.exists(model_info['model_path']):
            os.remove(model_info['model_path'])
else:
    logger.info("No accepted models found")

# Clean up temp directory if empty
if not os.listdir(temp_dir):
    os.rmdir(temp_dir)

logger.info(f"HTD experiment for {MODEL_NAME} completed")

max(results, key = lambda x: x['compression_rate'])

