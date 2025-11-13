import os
from collections import Counter

def check_class_balance(dataset_dir, dataset_name):
   
    # Check if the directory exists
    if not os.path.exists(dataset_dir):
        print(f"Error: Directory {dataset_dir} does not exist.")
        return
    
    # Get list of class folders
    class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
    
    if not class_folders:
        print(f"Error: No class folders found in {dataset_dir}.")
        return
    
    # Count images in each class
    class_counts = {}
    for class_name in class_folders:
        class_path = os.path.join(dataset_dir, class_name)
        # Count only image files with common extensions
        num_images = len([f for f in os.listdir(class_path) 
                         if os.path.isfile(os.path.join(class_path, f)) 
                         and f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = num_images
    
    # Print counts for each class
    print(f"\nNumber of images per class in {dataset_name}:")
    for class_name, count in sorted(class_counts.items()):
        print(f"{class_name}: {count}")
    
    # Check if all classes have the same number of images
    counts = list(class_counts.values())
    if len(set(counts)) == 1:
        print(f"All classes in {dataset_name} have the same number of images: {counts[0]}")
    else:
        print(f"Classes in {dataset_name} have different numbers of images.")
        print("Counts:", counts)
        print("Summary of counts:", Counter(counts))

# Dataset paths
dataset_paths = {
    'fruitsO': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\fruits_dataset\\fruitsO',
    'fruitsP': 'C:\\Users\\Admin\\Aruna_jrf_projects\\Datasets\\fruits_dataset\\fruitsP'
}

# Check balance for both datasets
for dataset_name, dataset_path in dataset_paths.items():
    check_class_balance(dataset_path, dataset_name)