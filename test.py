import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# Load digits dataset
digits = load_digits()
X = digits.images.reshape(-1, 64).astype('float32') / 16.0  # normalize to [0, 1]
y = digits.target

# Convert to torch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create TensorDatasets (already in memory)
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)


# In-memory DataLoaders (no on-the-fly transformation needed)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(train_dataset))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple FFN model
class SimpleFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Evaluation: negative loss as fitness
def evaluate_fitness(model, data_loader, device, quiet=False):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

    avg_loss = total_loss / total if total > 0 else float('inf')
    accuracy = correct / total if total > 0 else 0.0
    fitness = -avg_loss

    if not quiet:
        print(f"Eval - Accuracy: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}, Fitness: {fitness:.4f}")

    return fitness

# Full evaluation (just accuracy)
def full_evaluation(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total if total > 0 else 0.0

# Standard training loop using evaluate_fitness at the end
def train_and_evaluate(train_loader, val_loader, device, epochs=1000, quiet=False):
    model = SimpleFFN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if not quiet:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Final evaluation
    evaluate_fitness(model, val_loader, device, quiet=False)
    acc = full_evaluation(model, val_loader, device)
    print(f"Final Validation Accuracy: {acc:.4f}")
    return model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


import cProfile
cProfile.run("train_and_evaluate(train_loader, val_loader, device, 1000, quiet=True)", sort="time")
