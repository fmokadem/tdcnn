import torch
import torch.nn as nn
import time
import numpy as np
from ptflops import get_model_complexity_info
from torchsummary import summary
import copy
from tensorly.decomposition import partial_tucker
from scipy.linalg import svd
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker


def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, test_loader, device, num_runs=3):
    """Measure average inference time per image"""
    model.eval()
    inference_times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            for images, _ in test_loader:
                images = images.to(device)
                _ = model(images)
            end_time = time.time()
            inference_times.append((end_time - start_time) / len(test_loader))
    
    return np.mean(inference_times)

def calculate_accuracy(model, test_loader, device):
    """Calculate accuracy on the test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def get_flops(model, input_shape=(1, 3, 224, 224)):
    """
    Calculate FLOPs for a model
    
    Args:
        model: The model to analyze
        input_shape: Input shape in the format (batch_size, channels, height, width)
        
    Returns:
        flops: Number of floating-point operations
    """
    try:
        flops, params = get_model_complexity_info(
            model, 
            input_shape[1:], 
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False
        )
        return flops
    except Exception as e:
        print(f"Error calculating FLOPs: {e}")
        return 0  # Return 0 if there's an error

def is_conv2d_layer(module):
    """Check if a module is a Conv2d layer"""
    return isinstance(module, nn.Conv2d)

def get_conv2d_layers(model):
    """
    Get all Conv2d layers in a model with their names
    
    Returns:
        dict: {name: layer} for all Conv2d layers
    """
    conv_layers = {}
    
    def _get_conv_layers(module, prefix=''):
        for name, layer in module.named_children():
            layer_name = f"{prefix}.{name}" if prefix else name
            if is_conv2d_layer(layer):
                conv_layers[layer_name] = layer
            else:
                _get_conv_layers(layer, layer_name)
    
    _get_conv_layers(model)
    return conv_layers

def infer_rank(layer):
    """Infer initial rank for a Conv2d layer"""
    return min(layer.out_channels, layer.in_channels)

def calculate_layer_params(layer):
    """Calculate parameters in a Conv2d layer"""
    return layer.out_channels * layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]

def replace_conv2d_with_tucker(model, layer_name, layer, rank):
    """
    Replace a Conv2d layer with its Tucker decomposition
    
    Args:
        model: The model to modify
        layer_name: Name of the layer to replace
        layer: The Conv2d layer to replace
        rank: Rank to use for decomposition (tuple of ranks for output and input dimensions)
        
    Returns:
        model: Modified model with the layer replaced
    """
    
    # Navigate to the parent module containing the layer to replace
    parent_module = model
    name_parts = layer_name.split('.')
    
    # Navigate to the parent module
    for part in name_parts[:-1]:
        parent_module = getattr(parent_module, part)
    
    # Replace the layer with its decomposition
    decomposed_layer = tucker_decomp(layer, rank)
    setattr(parent_module, name_parts[-1], decomposed_layer)
    
    return model

def fine_tune(model, train_loader, device, epochs=3, lr=0.001):
    """
    Fine-tune a model for a specified number of epochs
    
    Args:
        model: Model to fine-tune
        train_loader: DataLoader for training data
        device: Device to train on
        epochs: Number of epochs to train
        lr: Learning rate
        
    Returns:
        model: Fine-tuned model
    """
    model.train()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")
    
    return model

def tucker_decomp(layer, rank):
    """
    Decompose a Conv2d layer into a Tucker decomposition
    
    Args:
        layer: The Conv2d layer to decompose
        rank: Rank to use for decomposition (tuple of ranks for output and input dimensions)
    
    Returns:
        sequential_layer: Sequential module with the Tucker decomposition
    """
    tl.set_backend('pytorch')
    # Extract the weight tensor from the original layer
    W = layer.weight.data

    # Perform partial Tucker decomposition on modes 0 and 1 (output and input channels)
    (core, [last, first]), rec_error = partial_tucker(W, modes=[0, 1], rank=rank, init='svd')

    # Define the first 1x1 convolution layer to transform input channels
    first_layer = nn.Conv2d(
        in_channels=first.shape[0],    # input_channels
        out_channels=first.shape[1],   # rank[1]
        kernel_size=1,
        padding=0,
        bias=False
    )

    # Define the core convolution layer with the original kernel size and attributes
    core_layer = nn.Conv2d(
        in_channels=core.shape[1],     # rank[1]
        out_channels=core.shape[0],    # rank[0]
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        dilation=layer.dilation,
        bias=False
    )

    # Define the last 1x1 convolution layer to transform to output channels
    last_layer = nn.Conv2d(
        in_channels=last.shape[1],     # rank[0]
        out_channels=last.shape[0],    # output_channels
        kernel_size=1,
        padding=0,
        bias=layer.bias is not None    # Bias only if original layer has bias
    )

    # Copy bias data to last_layer if the original layer has bias
    if layer.bias is not None:
        last_layer.bias.data = layer.bias.data

    # Prepare and assign weights
    fk = first.t_().unsqueeze_(-1).unsqueeze_(-1)  # Shape: [rank[1], input_channels, 1, 1]
    lk = last.unsqueeze_(-1).unsqueeze_(-1)        # Shape: [output_channels, rank[0], 1, 1]

    first_layer.weight.data = fk
    last_layer.weight.data = lk
    core_layer.weight.data = core                  # Shape: [rank[0], rank[1], kernel_height, kernel_width]

    # Create and return a sequential module
    sequential_layer = nn.Sequential(first_layer, core_layer, last_layer)
    return sequential_layer