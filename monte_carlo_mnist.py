import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import random
from torchvision import datasets, transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
# ---------------------------
# Data Generation and Preprocessing (MNIST)
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),                   # Convert PIL image to tensor in [0,1]
    transforms.Lambda(lambda x: x.view(-1))    # Flatten to 784-dim vector
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

X_train = train_dataset.data.float() / 255.0   # (60000, 28, 28)
X_test  = test_dataset.data.float() / 255.0      # (10000, 28, 28)
X_train = X_train.view(-1, 28*28)                # (60000, 784)
X_test  = X_test.view(-1, 28*28)                 # (10000, 784)
y_train = train_dataset.targets                 # (60000,)
y_test  = test_dataset.targets                  # (10000,)

# ---------------------------
# MC Dropout Model Definition with Feature Extraction
# ---------------------------
class MCDropoutNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128, 64], dropout_p=0.5, output_dim=10):
        """
        A fully connected network with dropout layers that remain active at test time.
        Note: The final layer now outputs raw logits (no Sigmoid).
        """
        super(MCDropoutNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_p))
            prev_dim = h_dim
        # Final output layer without activation (raw logits)
        self.hidden_net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        features = self.hidden_net(x)
        out = self.output_layer(features)
        return out
    
    def get_features(self, x):
        with torch.no_grad():
            return self.hidden_net(x)

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
# Training Function
# ---------------------------
def train_model(model, X_train, y_train, epochs=30, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == y_train).float().mean().item()
        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)
    return model, epoch_losses, epoch_accuracies

def train_mc_model(epochs=30, lr=0.001, input_dim=784, hidden_dims=[256,128,64], dropout_p=0.5):
    model = MCDropoutNN(input_dim=input_dim, hidden_dims=hidden_dims, dropout_p=dropout_p, output_dim=10)
    initialize_model(model, scheme='xavier_uniform')
    model, losses, accuracies = train_model(model, X_train, y_train, epochs=epochs, lr=lr)
    print(f"MC Dropout Model | Final Training Loss: {losses[-1]:.4f} | Final Training Accuracy: {accuracies[-1]:.4f}")
    return model, losses, accuracies

# ---------------------------
# Monte Carlo Dropout Prediction Function
# ---------------------------
def mc_dropout_predictions(model, X, T=100):
    """
    Perform T forward passes with dropout active to compute the mean prediction and variance.
    Apply softmax on the raw logits.
    """
    model.train()  # Ensure dropout is active at test time
    preds = []
    with torch.no_grad():
        for _ in range(T):
            preds.append(torch.softmax(model(X), dim=1).detach().numpy())
    preds = np.array(preds)  # (T, N, 10)
    preds_mean = preds.mean(axis=0)
    preds_variance = preds_mean * (1 - preds_mean)  # Shape: (N, 10)
    return preds_mean, preds_variance

def evaluate_mc_model(model, X, y, T=100, skip_nll=False):
    preds_mean, preds_variance = mc_dropout_predictions(model, X, T=T)
    pred_classes = preds_mean.argmax(axis=1)
    acc = accuracy_score(y.numpy(), pred_classes)
    nll = log_loss(y.numpy(), preds_mean) if not skip_nll else None
    return acc, nll, preds_mean, preds_variance

# ---------------------------
# Main Script: Train and Evaluate MC Dropout on MNIST
# ---------------------------
mc_model, train_losses, train_accuracies = train_mc_model(epochs=30, lr=0.001, input_dim=784, hidden_dims=[256,128,64], dropout_p=0.3)
T = 100
test_acc, test_nll, test_preds_mean, test_preds_variance = evaluate_mc_model(mc_model, X_test, y_test, T=T)
print(f"\nMC Dropout Evaluation | Test Accuracy: {test_acc:.4f} | Test NLL: {test_nll:.4f} | Mean Prediction Variance: {test_preds_variance.mean():.4f}")

# Compute per-sample uncertainty as the average variance over the 10 classes
sample_uncertainty = test_preds_variance.mean(axis=1)


#Project Features with PCA and Interpolate Uncertainty

from sklearn.manifold import TSNE

# Extract hidden features from the test set using the MC dropout model
features = mc_model.get_features(X_test)  # shape: (N, feature_dim)
features_np = features.detach().numpy()

print(f"Features Shape: {features_np.shape}")

# Run t-SNE to project features to 2D
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features_np)

# Create a grid over the t-SNE plane
xi = np.linspace(features_2d[:,0].min(), features_2d[:,0].max(), 200)
yi = np.linspace(features_2d[:,1].min(), features_2d[:,1].max(), 200)
xi, yi = np.meshgrid(xi, yi)

# Interpolate the uncertainty (sample_uncertainty) onto the grid
zi = griddata((features_2d[:,0], features_2d[:,1]), sample_uncertainty, (xi, yi), method='cubic')
zi = np.clip(zi, 0, None)  # Ensure non-negative values

# ---------------------------
# Plot the t-SNE Projection with Uncertainty Contours
# ---------------------------

plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:,0], features_2d[:,1], c=y_test.numpy(), cmap='jet', alpha=0.5)
plt.colorbar(label="Digit Label")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("t-SNE Projection of MNIST Test Features")
plt.grid(True)
plt.savefig('t-SNE.png')
plt.show()

plt.clf()

plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:,0], features_2d[:,1], c=sample_uncertainty, cmap='plasma', alpha=0.5)
plt.colorbar(label="Uncertainty (Prediction Variance)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("Uncertainty Distribution in t-SNE Space")
plt.grid(True)
plt.savefig('t-SNE_uncertainty_distribution.png')
plt.show()

plt.clf()