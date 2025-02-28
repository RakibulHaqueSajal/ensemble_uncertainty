import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.datasets import make_classification
import torch

def generate_classification_data(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, class_sep=1.0, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        random_state=random_state
    )
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# Example usage


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate 2D synthetic spiral dataset
def generate_spirals(n_points, noise=0):
    n = np.sqrt(np.random.rand(n_points)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.randn(n_points) * noise
    d1y = np.sin(n) * n + np.random.randn(n_points) * noise
    d2x = np.cos(n) * n + np.random.randn(n_points) * noise
    d2y = -np.sin(n) * n + np.random.randn(n_points) * noise
    X = np.vstack((np.column_stack((d1x, d1y)), np.column_stack((d2x, d2y))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)



# Neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 2 classes
        )
    
    def forward(self, x):
        return self.net(x)

# Training function with L2 regularization
def train_model(model, X_train, y_train, epochs=200, lr=0.01, weight_decay=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    return model

# Ensemble training and prediction
def train_ensemble(X_train, y_train, n_ensembles, seed_offset=2):
    models = []
    for i in range(n_ensembles):
        torch.manual_seed(i + seed_offset)
        model = SimpleNN()
        model = train_model(model, X_train, y_train)
        models.append(model)
    return models

# Evaluate ensemble predictions
def evaluate_ensemble(models, X_test):
    predictions = []
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(X_test)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.unsqueeze(0))
    predictions = torch.cat(predictions, dim=0)  # [n_ensembles, n_samples, n_classes]
    mean_probs = predictions.mean(dim=0)
    variance = predictions.var(dim=0).mean(dim=1)
    preds = mean_probs.argmax(dim=1)
    return preds, mean_probs, variance

# Generate dataset

X, y = generate_spirals(500)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Experiment with different ensemble sizes
ensemble_sizes = [3, 5, 7, 10]
results = {}

for n in ensemble_sizes:
    print(f"Training ensemble with {n} models...")
    models = train_ensemble(X_train, y_train, n_ensembles=n)
    preds, mean_probs, variance = evaluate_ensemble(models, X_test)
    accuracy = (preds == y_test).float().mean().item()
    avg_uncertainty = variance.mean().item()
    results[n] = {'accuracy': accuracy, 'uncertainty': avg_uncertainty}
    print(f"Ensemble size {n}: Accuracy = {accuracy:.4f}, Avg. Uncertainty = {avg_uncertainty:.4f}")

# Visualization of decision boundary and uncertainty for n=5
# New function to plot all ensemble sizes side by side
def plot_all_ensembles(X, y, ensemble_sizes):
    n_rows = 2  # Adjust based on the number of ensemble sizes
    n_cols = 2  # Adjust to fit all ensemble sizes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    for idx, n in enumerate(ensemble_sizes):
        models = train_ensemble(X_train, y_train, n_ensembles=n)
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        _, _, variance = evaluate_ensemble(models, grid)
        Z = variance.numpy().reshape(xx.shape)
        
        # Plot on the corresponding subplot
        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        axes[idx].scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
        axes[idx].set_title(f"Ensemble Size {n}")
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
    
    plt.tight_layout()
    plt.colorbar(axes[0].collections[0], ax=axes, label='Predictive Variance')
    plt.show()

# Use the function with your data
plot_all_ensembles(X, y, ensemble_sizes=[2, 3, 5, 10])

# Print results
print("\nSummary of Results:")
for n, res in results.items():
    print(f"Ensemble size {n}: Accuracy = {res['accuracy']:.4f}, Avg. Uncertainty = {res['uncertainty']:.4f}")