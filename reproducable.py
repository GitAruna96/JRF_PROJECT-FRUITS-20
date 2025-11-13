import os
import torch
import random
import numpy as np

def set_deterministic():
    """Set all random seeds for reproducibility"""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

def recover_78_45_experiment():
    """Recreate the exact conditions that gave 78.44%"""
    set_deterministic()
    
    # Exact parameters from your successful run (epoch 17)
    config = {
        'learning_rate': 1e-4,
        'lambda_sac': 1.0,
        'conf_start': 0.80,
        'conf_end': 0.95,
        'noise_scale': 0.05,
        'high_freq_ratio': 0.7,
        'batch_size': 32,
        'optimizer': 'AdamW',
        'weight_decay': 1e-4,
    }
    
    print("=== Recovering 78.45% Experiment ===")
    print("Configuration that worked:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    return config

# Run this first to see if we can reproduce
success_config = recover_78_45_experiment()