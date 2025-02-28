import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import random

# ---------------------------
# Data Generation and Preprocessing
# ---------------------------

# Generate a 2D synthetic dataset (moons dataset)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert training and test data to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

# ---------------------------
# Model and Initialization Functions
# ---------------------------

# New network with dropout layers added after each ReLU.
class MCDropoutNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[32, 16, 8], dropout_p=0.2, output_dim=1):
        """
        hidden_dims: list of integers specifying the number of nodes in each hidden layer.
        dropout_p: dropout probability (applied after each hidden layer).
        """
        super(MCDropoutNN, self).__init__()
        layers = []
        prev_dim = input_dim
        # Create hidden layers with dropout
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))  # Dropout layer
            prev_dim = h_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_model(model, X_train, y_train, epochs=50, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)   
    model.train()
    
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            preds = (output > 0.5).float()
            accuracy = (preds.eq(y_train)).float().mean().item()
        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)
    return model, epoch_losses, epoch_accuracies



def train_mc_model(epochs=50, lr=0.001, input_dim=2, hidden_dims=[32, 16, 8], dropout_p=0.2):
    # Create one model with MC Dropout layers.
    model = MCDropoutNN(input_dim=input_dim, hidden_dims=hidden_dims, dropout_p=dropout_p, output_dim=1)
    trained_model, losses, accuracies = train_model(model, X_train, y_train, epochs=epochs, lr=lr)
    print(f"Trained MC Dropout Model | Final Training Loss: {losses[-1]:.4f} | Final Training Accuracy: {accuracies[-1]:.4f}")
    return trained_model, losses, accuracies

# ---------------------------
# Monte Carlo Dropout Prediction Function
# ---------------------------

def mc_dropout_predictions(model, X, T=100):
    """
    Perform T stochastic forward passes (with dropout active) and return the mean and variance of the predictions.
    """
    model.train()  # Force dropout to be active
    preds = []
    with torch.no_grad():
        for _ in range(T):
            preds.append(model(X).detach().numpy())
    preds = np.array(preds)  # Shape: (T, N, 1)
    preds_mean = preds.mean(axis=0)
    preds_variance = preds.var(axis=0)
    return preds_mean, preds_variance

# ---------------------------
# Evaluation Functions
# ---------------------------

def evaluate_mc_model(model, X, y, T=100, skip_nll=False):
    preds_mean, preds_variance = mc_dropout_predictions(model, X, T=T)
    acc = accuracy_score(y, preds_mean > 0.5)
    nll = log_loss(y, preds_mean) if not skip_nll else None
    return acc, nll, preds_mean, preds_variance

# ---------------------------
# Main Script: Train MC Dropout Model and Evaluate
# ---------------------------

# Train a single MC Dropout model on the moons dataset
epochs = 100
mc_model, train_losses, train_accuracies = train_mc_model(epochs=epochs, lr=0.001, input_dim=2, hidden_dims=[32,16,8], dropout_p=0.2)

# Evaluate the MC Dropout model on the test set using T stochastic forward passes
T = 100  # Number of MC samples
test_acc, test_nll, test_preds_mean, test_preds_variance = evaluate_mc_model(mc_model, X_test, y_test, T=T)
print(f"\nMC Dropout Evaluation | Test Accuracy: {test_acc:.4f} | Test NLL: {test_nll:.4f} | Mean Prediction Variance: {test_preds_variance.mean():.4f}")

# ---------------------------
# Visualization: Uncertainty Contours via MC Dropout
# ---------------------------

# Create a grid over the test space for uncertainty visualization
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Get MC dropout predictions on the grid
grid_preds_mean, grid_preds_variance = mc_dropout_predictions(mc_model, grid, T=T)
grid_preds_variance = grid_preds_variance.reshape(xx.shape)

# Plot the uncertainty contours and test data
plt.figure(figsize=(10, 8))
contour = plt.contourf(xx, yy, grid_preds_variance, alpha=0.6, cmap='viridis')
plt.colorbar(contour, label='Prediction Variance')
plt.scatter(X_test[:, 0].numpy(), X_test[:, 1].numpy(), c=y_test.numpy().squeeze(), edgecolors='k', cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('MC Dropout Uncertainty Visualization')
plt.show()
