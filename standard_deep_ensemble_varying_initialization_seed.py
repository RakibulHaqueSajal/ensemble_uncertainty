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


# Generate a 2D synthetic dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)




# Example simple neural network (for binary classification)
class SimpleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Function to apply a chosen initialization scheme to a model
def initialize_model(model, scheme='xavier_uniform'):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == 'xavier_uniform':
                init.xavier_uniform_(m.weight)
            elif scheme == 'xavier_normal':
                init.xavier_normal_(m.weight)
            elif scheme == 'kaiming_normal':
                init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif scheme == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif scheme == 'orthogonal':
                init.orthogonal_(m.weight)
            else:
                raise ValueError(f"Unknown initialization scheme: {scheme}")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

# Example training function for a model
def train_model(model, X_train, y_train, epochs=50, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    return model

# Suppose we have some training data (replace with your own)
# For demonstration, we'll create some synthetic data.
X_train = torch.randn(500, 2)   # 500 samples, 2 features
y_train = (torch.sum(X_train, dim=1, keepdim=True) > 0).float()  # a simple binary target

# List of initialization schemes to choose from
init_schemes = ['xavier_uniform', 'xavier_normal', 'kaiming_normal', 'kaiming_uniform', 'orthogonal']

# Create an ensemble of models, each with a different (randomly chosen) initialization scheme
def train_ensemble(ensemble_size,epochs=50, lr=0.001, input_dim=2, hidden_dim=32):
    ensemble_models = []
    for i in range(ensemble_size):
        model = SimpleNN(input_dim=input_dim, hidden_dim=hidden_dim)
        chosen_scheme = random.choice(init_schemes)
        #print(f"Model {i+1} initialized with: {chosen_scheme}")
        initialize_model(model, scheme=chosen_scheme)
        trained_model = train_model(model, X_train, y_train, epochs=epochs, lr=lr)
        ensemble_models.append(trained_model)
    return ensemble_models


# Function to predict with an ensemble
def ensemble_predictions(models, X):
    preds = np.array([model(X).detach().numpy() for model in models])
    preds_mean = preds.mean(axis=0)
    preds_variance = preds.var(axis=0)
    return preds_mean, preds_variance

# Function to evaluate the ensemble
def evaluate_ensemble(models, X, y, skip_nll=False):
    preds_mean, preds_variance = ensemble_predictions(models, X)
    acc = accuracy_score(y, preds_mean > 0.5)
    nll = log_loss(y, preds_mean) if not skip_nll else None
    return acc, nll, preds_mean, preds_variance

# Training an ensemble of models with varying sizes
n_ensembles = [5, 10, 15, 20]
results = {}
for n in n_ensembles:
    models = train_ensemble(ensemble_size=n)
    acc, nll, preds_mean, preds_variance = evaluate_ensemble(models, X_test, y_test)
    results[n] = {'accuracy': acc, 'nll': nll, 'variance': preds_variance.mean()}
    print(f'Ensemble size: {n} | Accuracy: {acc:.4f} | NLL: {nll:.4f} | Variance: {preds_variance.mean():.4f}')

# Plotting uncertainty visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)


# Plot each ensemble size in a separate subplot
for i, n in enumerate(n_ensembles):
    ax = axes[i]
    models = train_ensemble(ensemble_size=n)
    _, _, _, preds_variance = evaluate_ensemble(models, grid, np.zeros(grid.shape[0]), skip_nll=True)
    
    # Reshape predictions to match grid shape
    preds_variance = preds_variance.reshape(xx.shape)
    
    # Contour plot in the respective subplot
    contour = ax.contourf(xx, yy, preds_variance, alpha=0.6, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Uncertainty (std)')
    
    # Scatter plot of the test data
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test.numpy(), edgecolors='k', cmap='viridis')
    
    # Set title and labels
    ax.set_title(f'Ensemble Size: {n}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

# Adjust layout for better visualization
plt.tight_layout()
plt.suptitle('Uncertainty Visualization for Different Ensemble Sizes', fontsize=16, y=1.02)
plt.show()
# Plotting ensemble uncertainties
ensemble_sizes = list(results.keys())
variances = [results[n]['variance'] for n in ensemble_sizes]

plt.figure(figsize=(8, 6))
plt.plot(ensemble_sizes, variances, marker='o')
plt.xlabel('Ensemble Size')
plt.ylabel('Average Prediction Variance')
plt.title('Ensemble Uncertainty vs. Ensemble Size')
plt.grid(True)
plt.show()