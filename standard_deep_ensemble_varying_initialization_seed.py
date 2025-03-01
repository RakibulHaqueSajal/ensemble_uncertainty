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

# Generate a 2D synthetic dataset for testing (moons dataset)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#convert training data to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Convert test data to tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ---------------------------
# Model and Initialization Functions
# ---------------------------

# Updated network that supports different number of nodes in each hidden layer.
class SimpleNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[32, 16, 8], output_dim=1):
        """
        hidden_dims: list of integers specifying the number of nodes in each hidden layer.
        """
        super(SimpleNN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers with varying sizes
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        # Use nn.Sequential to combine the layers
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

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

# ---------------------------
# Training and Evaluation Functions
# ---------------------------

def train_model(model, X_train, y_train, epochs=50, lr=0.001, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.0001)
    model.train()
    
    # Lists to store loss and accuracy over epochs for this model
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Compute training accuracy for this epoch
        with torch.no_grad():
            preds = (output > 0.5).float()
            accuracy = (preds.eq(y_train)).float().mean().item()
        
        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)
    
    return model, epoch_losses, epoch_accuracies

# List of initialization schemes
init_schemes = ['xavier_uniform', 'xavier_normal', 'kaiming_normal', 'kaiming_uniform', 'orthogonal']

def train_ensemble(ensemble_size, epochs=50, lr=0.001, input_dim=2, hidden_dims=[32, 16, 8]):
    ensemble_models = []
    training_metrics = []  # To store final training loss and accuracy for each model

    for i in range(ensemble_size):
        model = SimpleNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1)
        chosen_scheme = random.choice(init_schemes)
        # Uncomment the next line to see which initialization scheme is used.
        # print(f"Model {i+1} initialized with: {chosen_scheme}")
        initialize_model(model, scheme=chosen_scheme)
        trained_model, losses, accuracies = train_model(model, X_train, y_train, epochs=epochs, lr=lr, seed=i)
        ensemble_models.append(trained_model)
        training_metrics.append({'final_loss': losses[-1], 'final_accuracy': accuracies[-1]})
       # print(f'Model {i+1} | Init: {chosen_scheme} | Final Training Loss: {losses[-1]:.4f} | Final Training Accuracy: {accuracies[-1]:.4f}')
    return ensemble_models, training_metrics

def ensemble_predictions(models, X):
    preds = np.array([model(X).detach().numpy() for model in models])
    preds_mean = preds.mean(axis=0)
    preds_variance = preds_mean * (1 - preds_mean)  # Assuming binary classification
    return preds_mean, preds_variance

def evaluate_ensemble(models, X, y, skip_nll=False):
    preds_mean, preds_variance = ensemble_predictions(models, X)
    acc = accuracy_score(y, preds_mean > 0.5)
    nll = log_loss(y, preds_mean) if not skip_nll else None
    return acc, nll, preds_mean, preds_variance

# ---------------------------
# Main Script: Train Ensemble and Evaluate
# ---------------------------

# Evaluate different ensemble sizes
n_ensembles = [5, 10, 15, 20]
results = {}
ensemble_training_metrics = {}  # To store training metrics for each ensemble

# Here you can adjust the hidden_dims list for different numbers of nodes per layer
hidden_dims = [32, 16, 8]  # Example: first hidden layer 32 nodes, second 16, third 8

for n in n_ensembles:
    print(f'\nTraining ensemble with {n} models:')
    models, training_metrics = train_ensemble(ensemble_size=n, hidden_dims=hidden_dims)
    acc, nll, preds_mean, preds_variance = evaluate_ensemble(models, X_test, y_test)
    results[n] = {'accuracy': acc, 'nll': nll, 'variance': preds_variance.mean()}
    ensemble_training_metrics[n] = training_metrics
    print(f'--> Ensemble size: {n} | Test Accuracy: {acc:.4f} | Test NLL: {nll:.4f} | Mean Variance: {preds_variance.mean():.4f}')

# ---------------------------
# Compute and Print Average Training Metrics and Test Accuracy
# ---------------------------
print("\nAverage Training Metrics and Testing Accuracy for each Ensemble:")
for n in n_ensembles:
    metrics = ensemble_training_metrics[n]
    avg_loss = np.mean([m['final_loss'] for m in metrics])
    avg_acc = np.mean([m['final_accuracy'] for m in metrics])
    test_acc = results[n]['accuracy']
    print(f"Ensemble size: {n} | Average Training Loss: {avg_loss:.4f} | Average Training Accuracy: {avg_acc:.4f} | Test Accuracy: {test_acc:.4f}")

# ---------------------------
# Visualization: Uncertainty Contours on Test Domain
# ---------------------------

# Create a grid over the test space for uncertainty visualization
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, n in enumerate(n_ensembles):
    ax = axes[i]
    models, _ = train_ensemble(ensemble_size=n, hidden_dims=hidden_dims)
    
    # Compute uncertainty over grid points
    _, preds_variance = ensemble_predictions(models, grid)
    
    # Reshape variance to match grid shape
    preds_variance = preds_variance.reshape(xx.shape)
    
    # Contour plot for uncertainty
    contour = ax.contourf(xx, yy, preds_variance, alpha=0.6, cmap='viridis')
    fig.colorbar(contour, ax=ax, label='Bernoulli Uncertainty')

    # Scatter plot of test data
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test.numpy(), edgecolors='k', cmap='viridis')

    ax.set_title(f'Ensemble Size: {n}')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
plt.suptitle('Uncertainty Visualization for Different Ensemble Sizes', fontsize=16, y=1.02)
plt.show()

# ---------------------------
# Plotting Ensemble Uncertainty vs. Ensemble Size
# ---------------------------
ensemble_sizes = list(results.keys())
variances = [results[n]['variance'] for n in ensemble_sizes]

plt.figure(figsize=(8, 6))
plt.plot(ensemble_sizes, variances, marker='o')
plt.xlabel('Ensemble Size')
plt.ylabel('Average Prediction Variance')
plt.title('Ensemble Uncertainty vs. Ensemble Size')
plt.grid(True)
plt.show()
