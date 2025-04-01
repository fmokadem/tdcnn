import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import json
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import sys

# Create directories
plot_dir = './plots'
model_dir = './saved_models'
log_dir = './logs'
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def setup_logger(model_name):
    """Set up a custom logger for each model with both file and console handlers."""
    logger = logging.getLogger(f'model_training_{model_name}')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(message)s')
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{model_name}_training_{timestamp}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Function to load and preprocess the MNIST dataset
def load_mnist(data_dir='./data', batch_size=64, image_size=224):
    """Loads MNIST dataset with transformations for 3-channel 224x224 images."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels
        transforms.Resize((image_size, image_size)),  # Resize to 224x224
        transforms.ToTensor(),                        # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform)
    
    return train_dataset, test_dataset

# Function to plot training curves
def plot_training_curves(log_data, save_dir='./plots'):
    """Plot and save training curves from logged data."""
    os.makedirs(save_dir, exist_ok=True)
    model_name = log_data['model']
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(log_data['epochs'], log_data['train_losses'], 'b-', label='Training Loss')
    plt.title(f'{model_name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(log_data['epochs'], log_data['test_accuracies'], 'r-', label='Test Accuracy')
    plt.title(f'{model_name} Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_curves.png'))
    plt.close()

# Training function for a single model
def train_model(model_name, train_dataset, test_dataset, criterion, device, num_epochs=50):
    """Trains a specified model on a given GPU device."""
    # Set up logger for this model
    logger = setup_logger(model_name)
    logger.info(f"Initializing {model_name} training on {device}")
    
    # Initialize the model based on name
    if model_name == 'alexnet':
        model = models.alexnet(weights='DEFAULT')
        model.classifier[6] = nn.Linear(4096, 10)  # Adjust for 10 MNIST classes
    elif model_name == 'vgg16':
        model = models.vgg16(weights='DEFAULT')
        model.classifier[6] = nn.Linear(4096, 10)
    elif model_name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(512, 10)
    else:
        logger.error(f"Unknown model: {model_name}")
        raise ValueError(f"Unknown model: {model_name}")
    
    # Move model to the specified GPU
    model.to(device)
    logger.info(f"Model architecture:\n{model}")
    
    # Create data loaders inside the function to avoid multiprocessing issues
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    logger.info(f"Data loaders created with batch size {batch_size}")
    
    # Define optimizer with the same parameters as in the example
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    logger.info("Optimizer configured with lr=0.001, momentum=0.9, weight_decay=0.0005")
    
    # Initialize logging dictionary
    training_log = {
        "model": model_name,
        "train_losses": [],
        "test_accuracies": [],
        "epochs": [],
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "freeze_features": True
    }
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Log every 100 batches
            if batch_idx % 100 == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{num_epochs} completed - Average Loss: {avg_loss:.4f}")
        
        # Evaluation phase
        logger.info("Starting evaluation phase")
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        logger.info(f"Evaluation completed - Test Accuracy: {accuracy:.2f}%")
        
        # Update logging dictionary
        training_log["train_losses"].append(avg_loss)
        training_log["test_accuracies"].append(accuracy)
        training_log["epochs"].append(epoch + 1)
    
    # Save the trained model
    model_save_path = os.path.join(model_dir, f'{model_name}_mnist.pth')
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model saved to {model_save_path}")
    
    # Save training log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'{model_name}_training_log_{timestamp}.json')
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=4)
    logger.info(f"Training log saved to {log_path}")
    
    # Plot training curves
    plot_training_curves(training_log)
    logger.info(f"Training curves plotted and saved to ./plots/{model_name}_training_curves.png")
    logger.info("Training completed successfully!")

if __name__ == '__main__':
    
    # Set up root logger
    logging.basicConfig(level=logging.INFO)
    root_logger = logging.getLogger()
    root_logger.info("Starting training pipeline...")
    
    # Load datasets
    root_logger.info("Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist()
    root_logger.info("Dataset loaded successfully!")
    
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    
    # Define devices (assuming 3 GPUs are available)
    devices = ['cuda:0', 'cuda:1', 'cuda:2']
    root_logger.info(f"Using devices: {devices}")
    
    # List of models to train
    model_names = ['alexnet', 'vgg16', 'resnet18']
    root_logger.info(f"Models to train: {model_names}")
    
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Context is already set, proceed without raising an error
    
    # Create and start processes for each model
    processes = []
    root_logger.info("\nStarting parallel training processes...")
    for model_name, device in zip(model_names, devices):
        root_logger.info(f"Launching process for {model_name} on {device}")
        p = mp.Process(target=train_model, args=(model_name, train_dataset, test_dataset, criterion, device))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    root_logger.info("\nTraining of all models completed successfully!")
    root_logger.info("Check ./logs for detailed training logs")
    root_logger.info("Check ./plots for training curves")