import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import random
from torchvision import datasets, transforms

# ---------------------------
# Data Generation and Preprocessing (MNIST)
# ---------------------------
transform = transforms.Compose([
    transforms.ToTensor(),                   # Converts to [0,1] tensor with shape (1,28,28)
    transforms.Lambda(lambda x: x.view(-1))    # Flatten to 784-dim vector
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
X_train = train_dataset.data.float() / 255.0  # (60000, 28, 28)
X_test  = test_dataset.data.float() / 255.0   # (10000, 28, 28)
X_train = X_train.view(-1, 28*28)
X_test  = X_test.view(-1, 28*28)
y_train = train_dataset.targets
y_test  = test_dataset.targets

# ---------------------------
# Model and Initialization Functions
# ---------------------------
# Define a fully connected network. We use a separate module for hidden features.
class SimpleNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128, 64], output_dim=10):
        """
        hidden_dims: list of integers specifying the number of nodes in each hidden layer.
        For PCA visualization, we'll extract the features from the hidden network.
        """
        super(SimpleNN, self).__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        self.hidden_net = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
    def forward(self, x):
        hidden = self.hidden_net(x)
        out = self.output_layer(hidden)
        return out
    
    def get_features(self, x):
        # Return features from the hidden network.
        with torch.no_grad():
            features = self.hidden_net(x)
        return features

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
def train_model(model, X_train, y_train, epochs=10, lr=0.001):
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

init_schemes = ['xavier_uniform', 'xavier_normal', 'kaiming_normal', 'kaiming_uniform', 'orthogonal']
def train_ensemble(ensemble_size, epochs=10, lr=0.001, input_dim=784, hidden_dims=[256, 128, 64]):
    ensemble_models = []
    training_metrics = []
    for i in range(ensemble_size):
        model = SimpleNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10)
        chosen_scheme = random.choice(init_schemes)
        initialize_model(model, scheme=chosen_scheme)
        trained_model, losses, accuracies = train_model(model, X_train, y_train, epochs=epochs, lr=lr)
        ensemble_models.append(trained_model)
        training_metrics.append({'final_loss': losses[-1], 'final_accuracy': accuracies[-1]})
    return ensemble_models, training_metrics

def ensemble_predictions(models, X):
    preds = np.array([torch.softmax(model(X), dim=1).detach().numpy() for model in models])
    preds_mean = preds.mean(axis=0)
    preds_variance = preds.var(axis=0)
    return preds_mean, preds_variance

def evaluate_ensemble(models, X, y):
    preds_mean, preds_variance = ensemble_predictions(models, X)
    pred_classes = preds_mean.argmax(axis=1)
    acc = accuracy_score(y.numpy(), pred_classes)
    return acc, preds_mean, preds_variance

# ---------------------------
# Main Script: Train Ensemble and Compute Uncertainty
# ---------------------------
ensemble_size = 5
models, train_results= train_ensemble(ensemble_size=ensemble_size, epochs=10, lr=0.001, input_dim=784, hidden_dims=[256, 128, 64])
test_acc, preds_mean, preds_variance = evaluate_ensemble(models, X_test, y_test)
# Compute per-sample uncertainty: average variance across classes.
sample_uncertainty = preds_variance.mean(axis=1)
print(f"Ensemble Size {ensemble_size} | Test Accuracy: {test_acc:.4f} | Mean Uncertainty: {sample_uncertainty.mean():.4f}")
print(f"Training_Loss:{np.mean([r['final_loss'] for r in train_results])} | Training_Accuracy:{np.mean([r['final_accuracy'] for r in train_results])}")

# ---------------------------
# PCA Projection: Use the first two principal components of the hidden features
# ---------------------------
# Extract hidden features from the test set using one model (first model in ensemble)
model_for_features = models[0]
features = model_for_features.get_features(X_test)  # shape: (N, feature_dim)
features_np = features.detach().numpy()

# Run PCA to project features to 2D (using PC1 and PC2)
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features_np)

# ---------------------------
# Create a grid over the PCA space and interpolate uncertainty
# ---------------------------
xi = np.linspace(features_2d[:,0].min(), features_2d[:,0].max(), 200)
yi = np.linspace(features_2d[:,1].min(), features_2d[:,1].max(), 200)
xi, yi = np.meshgrid(xi, yi)

# Interpolate the uncertainty values onto the grid
zi = griddata((features_2d[:,0], features_2d[:,1]), sample_uncertainty, (xi, yi), method='cubic')
zi = np.clip(zi, 0, None)

# ---------------------------
# Plot the contourf of uncertainty on the PCA plane
# ---------------------------
plt.figure(figsize=(12,10))
contour = plt.contourf(xi, yi, zi, levels=20, cmap='viridis', alpha=0.6)
plt.colorbar(contour, label='Average Prediction Variance')

# Overlay the PCA points as a scatter plot, colored by their true class labels.
true_labels = y_test.numpy()
scatter = plt.scatter(features_2d[:,0], features_2d[:,1], c=true_labels, cmap='jet', edgecolors='k')
plt.title('PCA Projection of MNIST Test Features (PC1 vs PC2) with Uncertainty Contour')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
