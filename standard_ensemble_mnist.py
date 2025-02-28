import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import random
from torchvision import datasets, transforms

# ---------------------------
# Data Generation and Preprocessing (MNIST)
# ---------------------------

# Define a transform to convert PIL images to normalized tensors and flatten them
transform = transforms.Compose([
    transforms.ToTensor(),                   # Converts to [0,1] tensor with shape (1,28,28)
    transforms.Lambda(lambda x: x.view(-1))    # Flatten to vector of 784 elements
])

# Download and load the MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# For simplicity, we load all data into tensors (this may require sufficient RAM)
X_train = train_dataset.data.float() / 255.0  # shape: (60000, 28, 28)
X_test  = test_dataset.data.float() / 255.0   # shape: (10000, 28, 28)
# Flatten the images
X_train = X_train.view(-1, 28*28)
X_test  = X_test.view(-1, 28*28)
y_train = train_dataset.targets
y_test  = test_dataset.targets

# Convert targets to numpy arrays for evaluation later
y_train_np = y_train.numpy()
y_test_np  = y_test.numpy()

# Convert to torch tensors of appropriate type
X_train = X_train
X_test = X_test
y_train = y_train
y_test = y_test

# ---------------------------
# Model and Initialization Functions
# ---------------------------
# Updated network for MNIST: input_dim=784, adjustable hidden layers, output_dim=10.
class SimpleNN(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[256, 128, 64], output_dim=10):
        """
        hidden_dims: list of integers specifying the number of nodes in each hidden layer.
        """
        super(SimpleNN, self).__init__()
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers with ReLU activations
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        
        # Final output layer (logits for 10 classes)
        layers.append(nn.Linear(prev_dim, output_dim))
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
def train_model(model, X_train, y_train, epochs=10, lr=0.001):
    # For multi-class classification, use CrossEntropyLoss (which expects logits)
    criterion = nn.CrossEntropyLoss()
    # Using weight decay as regularization (optional)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    
    epoch_losses = []
    epoch_accuracies = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        # y_train contains class labels (0-9); CrossEntropyLoss expects raw logits and integer targets
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Compute training accuracy
        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == y_train).float().mean().item()
        epoch_losses.append(loss.item())
        epoch_accuracies.append(accuracy)
    return model, epoch_losses, epoch_accuracies

# List of initialization schemes
init_schemes = ['xavier_uniform', 'xavier_normal', 'kaiming_normal', 'kaiming_uniform', 'orthogonal']

def train_ensemble(ensemble_size, epochs=10, lr=0.001, input_dim=784, hidden_dims=[256, 128, 64]):
    ensemble_models = []
    training_metrics = []  # To store final training loss and accuracy for each model
    for i in range(ensemble_size):
        model = SimpleNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=10)
        chosen_scheme = random.choice(init_schemes)
        initialize_model(model, scheme=chosen_scheme)
        trained_model, losses, accuracies = train_model(model, X_train, y_train, epochs=epochs, lr=lr)
        ensemble_models.append(trained_model)
        training_metrics.append({'final_loss': losses[-1], 'final_accuracy': accuracies[-1]})
        # Uncomment below for detailed per-model output:
        # print(f'Model {i+1} | Init: {chosen_scheme} | Final Training Loss: {losses[-1]:.4f} | Final Training Accuracy: {accuracies[-1]:.4f}')
    return ensemble_models, training_metrics

def ensemble_predictions(models, X):
    # Get softmax probabilities from each model (we apply softmax over logits)
    preds = np.array([torch.softmax(model(X), dim=1).detach().numpy() for model in models])
    preds_mean = preds.mean(axis=0)         # Average probabilities
    preds_variance = preds.var(axis=0)        # Variance across ensemble predictions (per class)
    return preds_mean, preds_variance

def evaluate_ensemble(models, X, y):
    preds_mean, preds_variance = ensemble_predictions(models, X)
    # Predicted class is the argmax over averaged probabilities
    pred_classes = preds_mean.argmax(axis=1)
    acc = accuracy_score(y.numpy(), pred_classes)
    # Note: For multi-class, negative log likelihood can be computed if desired.
    # Here we report accuracy and also the average variance (as a rough uncertainty measure).
    return acc, preds_mean, preds_variance

# ---------------------------
# Main Script: Train Ensemble and Evaluate on MNIST
# ---------------------------
n_ensembles = [5]
results = {}
ensemble_training_metrics = {}  # To store training metrics for each ensemble

# Adjust hidden_dims for MNIST (input dim 784)
hidden_dims = [256, 128, 64]

for n in n_ensembles:
    print(f'\nTraining ensemble with {n} models:')
    models, training_metrics = train_ensemble(ensemble_size=n, epochs=10, lr=0.001, input_dim=784, hidden_dims=hidden_dims)
    test_acc, preds_mean, preds_variance = evaluate_ensemble(models, X_test, y_test)
    # For uncertainty, we compute the average variance across classes and examples.
    mean_variance = preds_variance.mean()
    results[n] = {'accuracy': test_acc, 'variance': mean_variance}
    ensemble_training_metrics[n] = training_metrics
    print(f'--> Ensemble size: {n} | Test Accuracy: {test_acc:.4f} | Mean Variance: {mean_variance:.4f}')

# ---------------------------
# Compute and Print Average Training Metrics and Test Accuracy for Each Ensemble
# ---------------------------
print("\nAverage Training Metrics and Testing Accuracy for each Ensemble:")
for n in n_ensembles:
    metrics = ensemble_training_metrics[n]
    avg_loss = np.mean([m['final_loss'] for m in metrics])
    avg_acc = np.mean([m['final_accuracy'] for m in metrics])
    test_acc = results[n]['accuracy']
    print(f"Ensemble size: {n} | Average Training Loss: {avg_loss:.4f} | Average Training Accuracy: {avg_acc:.4f} | Test Accuracy: {test_acc:.4f}")


# ---------------------------
# Visualization: Uncertainty Plots for Each Ensemble
# ---------------------------
# For each ensemble, we compute the uncertainty per test example (average variance across classes)
# and plot a histogram to visualize the distribution of uncertainties.
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, n in enumerate(n_ensembles):
    models, _ = train_ensemble(ensemble_size=n, epochs=10, lr=0.001, input_dim=784, hidden_dims=hidden_dims)
    _, _, preds_variance = evaluate_ensemble(models, X_test, y_test)
    # For each test example, compute the average variance across the 10 classes.
    sample_uncertainty = preds_variance.mean(axis=1)
    axes[i].hist(sample_uncertainty, bins=30, alpha=0.7, color='blue')
    axes[i].set_title(f'Ensemble Size: {n} Uncertainty Distribution')
    axes[i].set_xlabel('Average Prediction Variance')
    axes[i].set_ylabel('Frequency')

plt.tight_layout()
plt.suptitle('Uncertainty Distributions for Each Ensemble Size (MNIST)', fontsize=16, y=1.02)
plt.show()



# ---------------------------
# Visualization: Plot Ensemble Uncertainty vs. Ensemble Size
# ---------------------------
ensemble_sizes = list(results.keys())
variances = [results[n]['variance'] for n in ensemble_sizes]

plt.figure(figsize=(8, 6))
plt.plot(ensemble_sizes, variances, marker='o')
plt.xlabel('Ensemble Size')
plt.ylabel('Average Prediction Variance')
plt.title('Ensemble Uncertainty vs. Ensemble Size (MNIST)')
plt.grid(True)
plt.show()
