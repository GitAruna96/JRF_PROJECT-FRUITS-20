import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import InterpolationMode
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2 as transforms_v2
from torch.utils.data import DataLoader
import torch

def get_ssc_augmentation():
    """
    Returns a torchvision.transforms.v2 Compose object for SSC augmentation.
    This pipeline works on PIL Images and converts them to tensors.
    """
    ssc_augmentor = transforms.Compose([
        transforms.v2.ToImage(),
        transforms.v2.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.v2.RandomHorizontalFlip(p=0.5),
        transforms.v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.v2.RandomGrayscale(p=0.2),
        transforms.v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.v2.ToDtype(torch.float32, scale=True),
        transforms.v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return ssc_augmentor

def get_base_transform():
    """
    Returns the base transform for evaluation/test data.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class TwoViewImageFolder(datasets.ImageFolder):
    """
    A custom ImageFolder dataset that returns two augmented views of the same image.
    This is necessary for contrastive learning methods like SSC.
    """
    def __init__(self, root, transform, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.transform1 = get_ssc_augmentation()
        self.transform2 = get_ssc_augmentation()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (two_views, target) where two_views is a tuple of
                   two augmented image tensors and target is the class index.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        # Apply two different augmentations to the same image
        img1 = self.transform1(sample)
        img2 = self.transform2(sample)
        
        return (img1, img2), target

def get_loader(name_dataset, batch_size, train=True):
    dataset_paths = {
        'fruitsO': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\fruits_dataset\\fruitsO',
        'fruitsP': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\fruits_dataset\\fruitsP'
    }
    dataset_path = dataset_paths.get(name_dataset)
    if dataset_path is None:
        raise ValueError(f"Dataset {name_dataset} not found in defined paths")
    
    # Check if the dataset is for the target domain (fruitsP) and if it's training data
    is_trg_train = (name_dataset == 'fruitsP') and train
    
    if is_trg_train:
        # Use TwoViewImageFolder for the target training data
        dataset = TwoViewImageFolder(root=dataset_path, transform=None)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    elif train:
        # Use standard ImageFolder for the source training data with a single transformation
        transform = get_ssc_augmentation() # A single, strong augmentation
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    else:
        # For evaluation, use standard ImageFolder with a simple transform
        transform = get_base_transform()
        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    return loader