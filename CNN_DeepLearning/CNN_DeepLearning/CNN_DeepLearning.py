import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import struct
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Function to load MNIST images
def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        images = image_data.reshape(num_images, 1, rows, cols).astype(np.float32) / 255.0  # Normalize
        return images

# Function to load MNIST labels
def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Load dataset
train_images = load_mnist_images("train-images.idx3-ubyte")
train_labels = load_mnist_labels("train-labels.idx1-ubyte")
test_images = load_mnist_images("t10k-images.idx3-ubyte")
test_labels = load_mnist_labels("t10k-labels.idx1-ubyte")

# Convert to PyTorch tensors
train_images_tensor = torch.tensor(train_images)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
test_images_tensor = torch.tensor(test_images)
test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=0)  # kernel_size3x3, stride=1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool 2x2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0)  # kernel_size3x3
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer (16 channels, 5x5 image size)
        self.fc2 = nn.Linear(120, 84)  # Fully connected
        self.fc3 = nn.Linear(84, 10)  # Output layer
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 16 * 5 * 5)  # Flatten to 400
        x = F.relu(self.fc1(x))  # FC1 -> ReLU
        x = F.relu(self.fc2(x))  # FC2 -> ReLU
        x = self.fc3(x)  # Output layer (logits)
        return F.log_softmax(x, dim=1)

# Define CNN Model With 5 * 5 Kernal
class CNN5x5(nn.Module):
    def __init__(self):
        super(CNN5x5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)  # kernel_size5x5, padding=0
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool 2x2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # kernel_size5x5, padding=0
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Fully connected layer (16 channels, 4x4 image size)
        self.fc2 = nn.Linear(120, 84)  # Fully connected
        self.fc3 = nn.Linear(84, 10)  # Output layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = torch.flatten(x, 1)  # Flatten to 256
        x = F.relu(self.fc1(x))  # FC1 -> ReLU
        x = F.relu(self.fc2(x))  # FC2 -> ReLU
        x = self.fc3(x)  # Output layer (logits)
        return F.log_softmax(x, dim=1)

class CNN5x5_Modified(nn.Module):
    def __init__(self):
        super(CNN5x5_Modified, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, padding=0)  # (1, 8)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # MaxPool 2x2
        self.conv2 = nn.Conv2d(8, 32, kernel_size=5, stride=1, padding=0)  # (8, 32)
        self.fc1 = nn.Linear(32 * 4 * 4, 120)  # Adjusted for new feature map size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = torch.flatten(x, 1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# Training Loop
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    epoch_losses = []  # List to store loss for each epoch
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(train_loader))  # Append loss for this epoch
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    return epoch_losses  # Return loss for all epochs

# Testing Loop
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Set device to accelerate training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set Loss Function
criterion = nn.CrossEntropyLoss()
epochs = 5
# Initialize and train models
print("\nTraining CNN3x3:")
model_3x3 = CNN().to(device)
optimizer_3x3 = optim.Adam(model_3x3.parameters(), lr=0.001)  # Reinitialize optimizer
loss_3x3 = train(model_3x3, train_loader, criterion, optimizer_3x3, epochs)
test(model_3x3, test_loader)

print("\nTraining CNN5x5:")
model_5x5 = CNN5x5().to(device)
optimizer_5x5 = optim.Adam(model_5x5.parameters(), lr=0.001)  # Reinitialize optimizer
loss_5x5 = train(model_5x5, train_loader, criterion, optimizer_5x5, epochs)
test(model_5x5, test_loader)

print("\nTraining CNN5x5_Modified:")
model_5x5_Modified = CNN5x5_Modified().to(device)
optimizer_5x5_Modified = optim.Adam(model_5x5_Modified.parameters(), lr=0.001)  # Reinitialize optimizer
loss_5x5_Modified = train(model_5x5_Modified, train_loader, criterion, optimizer_5x5_Modified, epochs)
test(model_5x5_Modified, test_loader)

# Plot the losses for both models
plt.plot(range(1, epochs+1), loss_3x3, label="CNN3x3 Loss")
plt.plot(range(1, epochs+1), loss_5x5, label="CNN5x5 Loss")
plt.plot(range(1, epochs+1), loss_5x5_Modified, label="CNN5x5_Modified Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Epochs vs Loss")
plt.legend()
plt.show()