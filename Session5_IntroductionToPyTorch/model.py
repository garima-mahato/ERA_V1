import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    """This defines the structure of the NN.

    Args:
        nn (_type_): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self):
        """_summary_
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3) 
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3) 
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = F.relu(self.conv1(x))                # input_size = 28x28x1, output_size = 26x26x32, RF = 3x3
        x = F.relu(F.max_pool2d(self.conv2(x),2))  # input_size = 26x26x32, output_size = 12x12x64, RF = 6x6
        x = F.relu(self.conv3(x))                # input_size = 12x12x64, output_size = 10x10x128, RF = 10x10
        x = F.relu(F.max_pool2d(self.conv4(x),2))  # input_size = 10x10x128, output_size = 4x4x256, RF = 16x16
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)