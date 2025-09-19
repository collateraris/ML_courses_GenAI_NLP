import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Tdata
import torch.nn.functional as F
import torch.distributions as TD

from tqdm.notebook import tqdm

if torch.cuda.is_available():
    DEVICE = 'cuda'
    # GPU_DEVICE = 2
    # torch.cuda.set_device(GPU_DEVICE)
else:
    DEVICE='cpu'

import warnings
warnings.filterwarnings('ignore')

# typing
from typing import List, Tuple

from dgm_utils import plot_training_curves
from dgm_utils import show_samples, visualize_images, load_dataset

train_data_bin, test_data_bin = load_dataset("mnist", flatten=False, binarize=True)
visualize_images(train_data_bin, "Binarized MNIST samples")
train_data, test_data = load_dataset("mnist", flatten=False, binarize=False)
visualize_images(train_data, "MNIST samples")
