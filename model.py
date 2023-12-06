import torch.nn as nn
import torch.nn.functional as F


# nn architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First fully connected layer
        self.fc1 = nn.Linear(6, 12)
        # Second fully connected layer
        #self.fc2 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(12, 12)
        # Output layer, 3 classes
        self.fc3 = nn.Linear(12, 3)

    def forward(self, x):
        # ReLU activation function after first layer
        x = F.relu(self.fc1(x))
        # ReLU activation function after second layer
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        # No activation for out
        x = self.fc3(x)
        return x