import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def get_loader(name_dataset, batch_size, train=True, domain = 'source'):
    dataset_paths = {
        'fruitsO': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\fruits_dataset\\fruitsO',
        'fruitsP': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\fruits_dataset\\fruitsP'
    }
    if train:
        if domain =='source': #strong augmentation
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale = (0.8,1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:  #weak augmentation for trg domain
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    else:  #for bothsrc and trg domain for evaluation
        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        
    dataset_path = dataset_paths.get(name_dataset)
    if dataset_path is None:
        raise ValueError(f"Dataset {name_dataset} not found in defined paths")
    
    dataset = datasets.ImageFolder(root = dataset_path, transform = transform)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = train, num_workers = 4, pin_memory=True, drop_last = train)
    
    return loader