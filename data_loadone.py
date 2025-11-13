import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch
import os

def get_loader(name_dataset, batch_size, train=True):
    print("Using get_loader from data_load.py")  # Debug to confirm correct function
    dataset_paths = {
        'fruitsO': "C:\\Users\\Admin\\Desktop\\Aruna folder\\fruits\\fruitsO",
        'fruitsP': "C:\\Users\\Admin\\Desktop\\Aruna folder\\fruits\\fruitsP"
    }
    
    dataset_path = dataset_paths.get(name_dataset)
    if not dataset_path or not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path not found: {dataset_path}")
    
    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.RandomHorizontalFlip(p=0.5),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    try:
        print(f"Loading dataset from: {dataset_path}")
        dataset = ImageFolder(root=dataset_path, transform=transform)
        print(f"Dataset loaded successfully: {len(dataset)} samples, {len(dataset.classes)} classes")
        
        # Debug: Check a sample from the dataset
        sample_img, sample_label = dataset[0]
        print(f"Sample image type: {type(sample_img)}, shape: {sample_img.shape}, label type: {type(sample_label)}, label: {sample_label}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0, collate_fn=default_collate)
    
    # Debug: Check the first batch
    for X, y in loader:
        print(f"DataLoader batch X type: {type(X)}, shape: {X.shape if isinstance(X, torch.Tensor) else [t.shape for t in X]}")
        print(f"DataLoader batch y type: {type(y)}, shape: {y.shape}")
        break
    
    return loader