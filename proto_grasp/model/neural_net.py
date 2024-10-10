import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, config):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(config.input_size, config.hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(config.hidden_size, config.num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out