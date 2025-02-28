import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import torch
import torch.nn as nn
import torch.optim as optim


# Generate a 2D synthetic dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define a simple neural network with L2 regularization
class SimpleNN(nn.Module):
    def __init__(self, l2_reg=1e-4):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.l2_reg = l2_reg
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Function to train an ensemble of models with random seeds
def train_ensemble(n_models=5, l2_reg=1e-4, epochs=50, batch_size=32):
    models = []
    for i in range(n_models):
       # print(f'Training model {i+1}/{n_models}...')
        torch.manual_seed(np.random.randint(0, 10000))
        model = SimpleNN(l2_reg=l2_reg)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), weight_decay=l2_reg)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
        models.append(model)
    return models

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
    models = train_ensemble(n_models=n)
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
    models = train_ensemble(n_models=n)
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