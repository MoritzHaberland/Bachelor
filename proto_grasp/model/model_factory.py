import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from proto_grasp.model.neural_net import NeuralNet

# Model factory to handle multiple models
class ModelFactory:
    def __init__(self):
        self.models = {
            'neural_net': NeuralNet,  
        }

    def get_model(self, model_name, model_config):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        return self.models[model_name](model_config)