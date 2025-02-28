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
# Use the moons dataset for both training and testing.
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32).view(-1, 1)

# ---------------------------
# Concrete Dropout Module
# ---------------------------
class ConcreteDropout(nn.Module):
    def __init__(self, layer, weight_regularizer, dropout_regularizer, temperature=0.1, init_min=0.1, init_max=0.1):
        """
        Wraps a given layer (e.g. nn.Linear) with Concrete Dropout.
        
        weight_regularizer: scales the weight penalty.
        dropout_regularizer: scales the dropout penalty.
        temperature: temperature for the Concrete distribution.
        init_min, init_max: initial dropout probability range.
        """
        super(ConcreteDropout, self).__init__()
        self.layer = layer
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.temperature = temperature
        # Initialize logit for dropout probability p using a uniform init on the logit space.
        init_p = random.uniform(init_min, init_max)
        # logit_p = log(p) - log(1-p)
        self.logit_p = nn.Parameter(torch.log(torch.tensor(init_p)) - torch.log(torch.tensor(1.0 - init_p)))
        self.regularization = 0

    def forward(self, x):
        # Compute dropout probability
        p = torch.sigmoid(self.logit_p)
        eps = 1e-7
        # Sample dropout mask using Concrete distribution:
        # Sample u ~ Uniform(0,1) of same shape as input
        u = torch.rand_like(x)
        # Compute the Concrete dropout probability
        drop_prob = torch.sigmoid((torch.log(p + eps) - torch.log(1 - p + eps) + torch.log(u + eps) - torch.log(1 - u + eps)) / self.temperature)
        # Create dropout mask and scale
        dropout_mask = 1 - drop_prob
        x_dropped = x * dropout_mask / (1 - p)
        # Pass through the wrapped layer
        out = self.layer(x_dropped)
        
        # Compute regularization terms:
        # Weight regularization: weight_reg * ||W||^2 / (1-p)
        weight_reg = self.weight_regularizer * torch.sum(self.layer.weight ** 2) / (1 - p)
        # Dropout regularization: p * log(p) + (1-p)*log(1-p) times the number of inputs.
        dropout_reg = self.dropout_regularizer * x.numel() * (p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        
        self.regularization = weight_reg + dropout_reg
        return out

# ---------------------------
# Concrete Dropout Network
# ---------------------------
class ConcreteDropoutNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[32, 16, 8], output_dim=1,
                 weight_regularizer=1e-6, dropout_regularizer=1e-5, temperature=0.1, dropout_init=0.1):
        """
        Build a network with Concrete Dropout applied to each hidden linear layer.
        """
        super(ConcreteDropoutNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim
        # Create hidden layers: wrap each Linear layer with ConcreteDropout.
        for h_dim in hidden_dims:
            linear = nn.Linear(prev_dim, h_dim)
            cd_layer = ConcreteDropout(linear, weight_regularizer, dropout_regularizer, temperature=temperature, init_min=dropout_init, init_max=dropout_init)
            self.hidden_layers.append(cd_layer)
            # Add a ReLU activation after dropout (not wrapped)
            self.hidden_layers.append(nn.ReLU())
            prev_dim = h_dim
        # Final output layer (no dropout)
        self.out_layer = nn.Linear(prev_dim, output_dim)
        self.out_activation = nn.Sigmoid()
    
    def forward(self, x):
        reg_loss = 0
        for layer in self.hidden_layers:
            x = layer(x)
            if hasattr(layer, 'regularization'):
                reg_loss += layer.regularization
        out = self.out_layer(x)
        out = self.out_activation(out)
        # Store regularization loss in the module (to be added to training loss)
        self.regularization_loss = reg_loss
        return out

# ---------------------------
# Training Functions
# ---------------------------
def train_model(model, X_train, y_train, epochs=100, lr=0.001):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_train)
        # Standard loss
        loss = criterion(output, y_train)
        # Add Concrete Dropout regularization loss
        loss += model.regularization_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            preds = (output > 0.5).float()
            accuracy = (preds.eq(y_train)).float().mean().item()
        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)
    return model, epoch_losses, epoch_accuracies

# ---------------------------
# Monte Carlo Prediction Function for Concrete Dropout
# ---------------------------
def mc_dropout_predictions(model, X, T=100):
    """
    Perform T stochastic forward passes (with Concrete Dropout active).
    Note: In Concrete Dropout, we want dropout to be active at test time to sample uncertainty.
    """
    model.train()  # Keep dropout active
    preds = []
    with torch.no_grad():
        for _ in range(T):
            preds.append(model(X).detach().numpy())
    preds = np.array(preds)  # Shape: (T, N, 1)
    preds_mean = preds.mean(axis=0)
    preds_variance = preds.var(axis=0)
    return preds_mean, preds_variance

def evaluate_model(model, X, y, T=100, skip_nll=False):
    preds_mean, preds_variance = mc_dropout_predictions(model, X, T=T)
    acc = accuracy_score(y, preds_mean > 0.5)
    nll = log_loss(y, preds_mean) if not skip_nll else None
    return acc, nll, preds_mean, preds_variance

# ---------------------------
# Main Script: Train and Evaluate Concrete Dropout Model
# ---------------------------
# Hyperparameters for Concrete Dropout:
weight_reg = 1e-6
dropout_reg = 1e-5
temperature = 0.1
dropout_init = 0.1  # initial dropout probability

# Build the network
model = ConcreteDropoutNN(input_dim=2, hidden_dims=[32, 16, 8], output_dim=1,
                          weight_regularizer=weight_reg, dropout_regularizer=dropout_reg,
                          temperature=temperature, dropout_init=dropout_init)

# Train the model
epochs = 100
model, train_losses, train_accuracies = train_model(model, X_train, y_train, epochs=epochs, lr=0.001)
print(f"Final Training Loss: {train_losses[-1]:.4f} | Final Training Accuracy: {train_accuracies[-1]:.4f}")

# Evaluate using Monte Carlo Concrete Dropout predictions
T = 100  # number of stochastic forward passes
test_acc, test_nll, test_preds_mean, test_preds_variance = evaluate_model(model, X_test, y_test, T=T)
print(f"\nConcrete Dropout Evaluation | Test Accuracy: {test_acc:.4f} | Test NLL: {test_nll:.4f} | Mean Prediction Variance: {test_preds_variance.mean():.4f}")

# ---------------------------
# Visualization: Uncertainty Contours via Concrete Dropout
# ---------------------------
# Create a grid over the test space
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

grid_preds_mean, grid_preds_variance = mc_dropout_predictions(model, grid, T=T)
grid_preds_variance = grid_preds_variance.reshape(xx.shape)

plt.figure(figsize=(10, 8))
contour = plt.contourf(xx, yy, grid_preds_variance, alpha=0.6, cmap='viridis')
plt.colorbar(contour, label='Prediction Variance')
plt.scatter(X_test[:, 0].numpy(), X_test[:, 1].numpy(), c=y_test.numpy().squeeze(), edgecolors='k', cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Concrete Dropout Uncertainty Visualization')
plt.show()
