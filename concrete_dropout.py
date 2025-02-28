import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data (binary classification)
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----------------------------
# Concrete Dropout Module
# ----------------------------
class ConcreteDropout(nn.Module):
    def __init__(self, layer, weight_regularizer, dropout_regularizer, init_dropout=0.1, temperature=0.1):
        """
        Wraps a layer with Concrete Dropout.

        Args:
            layer: Underlying layer (e.g., nn.Linear).
            weight_regularizer: Coefficient for weight regularization.
            dropout_regularizer: Coefficient for dropout regularization.
            init_dropout: Initial dropout probability.
            temperature: Temperature parameter for the Concrete distribution.
        """
        super(ConcreteDropout, self).__init__()
        self.layer = layer
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.temperature = temperature
        # Initialize dropout probability in logit space for learnability
        self.logit_p = nn.Parameter(torch.log(torch.tensor(init_dropout)) - torch.log(torch.tensor(1.0 - init_dropout)))

    def forward(self, x):
        p = torch.sigmoid(self.logit_p)
        eps = 1e-7
        # Sample uniform noise
        unif_noise = torch.rand_like(x)
        # Compute dropout probability using the Concrete (Gumbel-Softmax) relaxation
        drop_prob = (torch.log(p + eps) - torch.log(1 - p + eps) +
                     torch.log(unif_noise + eps) - torch.log(1 - unif_noise + eps))
        drop_prob = torch.sigmoid(drop_prob / self.temperature)
        # Create dropout mask and scale the input to maintain expectation
        random_tensor = 1 - drop_prob
        x = x * random_tensor / (1 - p)
        return self.layer(x)

    def regularization_loss(self):
        p = torch.sigmoid(self.logit_p)
        weight = self.layer.weight
        # Weight regularization: scales inversely with (1 - dropout probability)
        weight_reg = self.weight_regularizer * torch.sum(weight ** 2) / (1 - p)
        # Dropout regularization encourages p not to become too high or too low
        dropout_reg = self.dropout_regularizer * (p * torch.log(p + 1e-7) +
                                                    (1 - p) * torch.log(1 - p + 1e-7)) * weight.numel()
        return weight_reg + dropout_reg

# ----------------------------
# Define the Network with Concrete Dropout
# ----------------------------
class ConcreteDropoutNet(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super(ConcreteDropoutNet, self).__init__()
        self.cd1 = ConcreteDropout(nn.Linear(input_dim, hidden_dim), weight_regularizer, dropout_regularizer)
        self.cd2 = ConcreteDropout(nn.Linear(hidden_dim, hidden_dim), weight_regularizer, dropout_regularizer)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Final layer without dropout

    def forward(self, x):
        x = F.elu(self.cd1(x))  # Using ELU activation
        x = F.elu(self.cd2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def regularization_loss(self):
        return self.cd1.regularization_loss() + self.cd2.regularization_loss()

# ----------------------------
# Training Function
# ----------------------------
def train_concrete_dropout_model(model, X_train, y_train, epochs=100, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        bce_loss = criterion(outputs, y_train)
        reg_loss = model.regularization_loss()
        loss = bce_loss + reg_loss
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Total Loss: {loss.item():.4f}, BCE: {bce_loss.item():.4f}, Reg: {reg_loss.item():.4f}")
    return model

# ----------------------------
# Monte Carlo Sampling for Concrete Dropout
# ----------------------------
def mc_concrete_predict(model, X, num_samples=100):
    # Set model to training mode to activate dropout
    model.train()
    predictions = []
    with torch.no_grad():
        for _ in range(num_samples):
            predictions.append(model(X))
    predictions = torch.stack(predictions)  # Shape: [num_samples, N, 1]
    mean_pred = predictions.mean(dim=0)
    var_pred = predictions.var(dim=0)
    return mean_pred, var_pred

# ----------------------------
# Train the Model
# ----------------------------
model_cd = ConcreteDropoutNet(input_dim=2, hidden_dim=32)
trained_model_cd = train_concrete_dropout_model(model_cd, X_train, y_train, epochs=100, lr=0.01)

# Evaluate on test set
trained_model_cd.eval()
with torch.no_grad():
    test_preds = (trained_model_cd(X_test) > 0.5).float()
print("Test Accuracy:", accuracy_score(y_test.numpy(), test_preds.numpy()))

# ----------------------------
# Uncertainty Visualization on a Grid
# ----------------------------
# Create a grid over the input space
x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Obtain Concrete Dropout predictions on the grid via Monte Carlo sampling
mean_preds_grid, var_preds_grid = mc_concrete_predict(trained_model_cd, grid, num_samples=100)
# Reshape variance predictions to grid shape
var_grid = var_preds_grid.reshape(xx.shape).detach().numpy()

# Plot the predictive uncertainty as a contour plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(xx, yy, var_grid, cmap='viridis', alpha=0.8)
plt.colorbar(contour, label='Prediction Variance')
plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=y.numpy().squeeze(),
            edgecolor='k', cmap='coolwarm', alpha=0.6)
plt.title('Concrete Dropout Uncertainty (Variance) Contour Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
