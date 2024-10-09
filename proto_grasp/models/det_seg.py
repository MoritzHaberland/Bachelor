import torch.nn as nn

class DetSeg(nn.Module):
    def __init__(self):
        super(self).__init()
        self.lin = nn.Linear(2, 1)

    def forward(self,x):
        return self.lin(x)