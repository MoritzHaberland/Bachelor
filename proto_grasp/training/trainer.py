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
        self.model.train()  # Set the module to training mode
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for inputs, targets in self.train_loader:
                self.optimizer.zero_grad()  # Zero the parameter gradients

                # Forward pass
                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, targets)  # Calculate the loss

                # Backward pass
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights

                running_loss += loss.item()  # Accumulate the loss

            # Print the average loss for this epoch
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {running_loss / len(self.train_loader):.4f}')

