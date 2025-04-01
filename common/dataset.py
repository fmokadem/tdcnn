import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



# def load_mnist(data_dir='/home/fmokadem/NAS/data', batch_size=64, image_size=224):
#     """
#     Load MNIST dataset with standard preprocessing.
    
#     Args:
#         data_dir: Directory where MNIST data is stored
#         batch_size: Batch size for data loaders
#         image_size: Size to resize images to (e.g., 224 for standard CNN models)
        
#     Returns:
#         train_loader, test_loader: DataLoader objects for train and test sets
#     """
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((image_size, image_size), antialias=True),
#         transforms.Normalize((0.1307,), (0.3081,))
#     ])
    
#     # Load training data
#     train_dataset = datasets.MNIST(
#         root=data_dir,
#         train=True,
#         download=True,
#         transform=transform
#     )
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=True
#     )
    
#     # Load test data
#     test_dataset = datasets.MNIST(
#         root=data_dir,
#         train=False,
#         download=True,
#         transform=transform
#     )
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=batch_size,
#         shuffle=False
#     )
    
#     return train_loader, test_loader 