import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Define the data transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='/hpc2hdd/home/ydingaz/workspace/Falcon/cifar_data', train=True, download=False, transform=transform)
print("download finish")
# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Load a pre-trained ResNet model
model = models.resnet101(weights='DEFAULT')  # You can use ResNet-18 or other pre-trained models

# Modify the last fully connected layer for CIFAR-100 (100 classes)
num_classes = 100
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
epoch = 0
while True:
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    epoch+=1 
    print(f'Epoch [{epoch + 1}], Loss: {running_loss / len(train_loader):.4f}')