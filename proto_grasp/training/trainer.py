import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, train_loader: DataLoader, num_epochs: int):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.num_epochs = num_epochs

    def train(self):
        """Train the neural network module."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        n_total_steps = len(self.train_loader)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(self.train_loader):  
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                images = images.reshape(-1, 28*28).to(device)
                labels = labels.to(device)
                
                # Forward pass and loss calculation
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward and optimize
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                if (i+1) % 100 == 0:
                    print (f'Epoch [{epoch+1}/{self.num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')