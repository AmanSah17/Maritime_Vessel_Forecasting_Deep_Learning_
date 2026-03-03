import random
import os
import yaml
import numpy as np
import torch

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load yaml configuration."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
