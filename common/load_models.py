import torch
from torchvision import models
import os
import torch.nn as nn

def load_model(model_name, model_path, device):
    """
    Load a pre-trained model from model_path, and adapt it to mnist dataset
    
    Args:
        model_name (str): Name of the model to load. 
                         Options: 'vgg', 'resnet', 'alexnet'
        model_path (str): Path to the model file
        device (str): Device to load the model to
                         
    Returns:
        torch.nn.Module: The loaded PyTorch model
    """
    # Check if the model name is valid
    valid_models = ['vgg', 'resnet', 'alexnet']
    if model_name not in valid_models:
        raise ValueError(f"Model '{model_name}' not supported. Choose from: {valid_models}")
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found")

    try:
        # First try to load the entire model
        saved_model = torch.load(model_path, map_location=device)
        
        # Initialize the base model with RGB input (3 channels)
        if model_name == 'vgg':
            model = models.vgg16(weights=None)
            model.classifier[6] = nn.Linear(4096, 10)
            
        elif model_name == 'alexnet':
            model = models.alexnet(weights=None)
            model.classifier[6] = nn.Linear(4096, 10)
            
        elif model_name == 'resnet':
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(512, 10)
        
        # Load the state dict
        if isinstance(saved_model, dict):
            model.load_state_dict(saved_model)
        else:
            model.load_state_dict(saved_model.state_dict())
        
        # Now modify the first layer to handle grayscale input
        # if model_name == 'vgg':
        #     old_layer = model.features[0]
        #     new_layer = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        #     # Average the weights across input channels
        #     with torch.no_grad():
        #         new_layer.weight.data = old_layer.weight.data.sum(dim=1, keepdim=True) / 3.0
        #         if old_layer.bias is not None:
        #             new_layer.bias.data = old_layer.bias.data
        #     model.features[0] = new_layer
            
        # elif model_name == 'alexnet':
        #     old_layer = model.features[0]
        #     new_layer = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        #     # Average the weights across input channels
        #     with torch.no_grad():
        #         new_layer.weight.data = old_layer.weight.data.sum(dim=1, keepdim=True) / 3.0
        #         if old_layer.bias is not None:
        #             new_layer.bias.data = old_layer.bias.data
        #     model.features[0] = new_layer
            
        # elif model_name == 'resnet':
        #     old_layer = model.conv1
        #     new_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #     # Average the weights across input channels
        #     with torch.no_grad():
        #         new_layer.weight.data = old_layer.weight.data.sum(dim=1, keepdim=True) / 3.0
        #     model.conv1 = new_layer
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # For testing, let's just try AlexNet since that's what we have
    try:
        print("\nLoading alexnet model...")
        model = load_model('alexnet', '/home/fmokadem/NAS/models/saved_models/alexnet_mnist.pth', device)
        print(f"Successfully loaded alexnet model")
        print(f"Model is on device: {next(model.parameters()).device}")
        
        # Print model structure to verify
        print("\nModel structure:")
        print(f"Input layer: {model.features[0]}")
        print(f"Output layer: {model.classifier[6]}")
        
        # Verify input channel count
        print(f"\nInput channel count: {model.features[0].weight.shape[1]}")
    except Exception as e:
        print(f"Error loading alexnet model: {e}\n") 