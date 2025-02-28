import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate 2D synthetic spiral dataset
def generate_spirals(n_points, noise=0.2):
    n = np.sqrt(np.random.rand(n_points)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.randn(n_points) * noise
    d1y = np.sin(n) * n + np.random.randn(n_points) * noise
    d2x = np.cos(n) * n + np.random.randn(n_points) * noise
    d2y = -np.sin(n) * n + np.random.randn(n_points) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((d2x, d2y))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)