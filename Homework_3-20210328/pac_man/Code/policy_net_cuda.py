from torch import nn
from torch.nn import functional as F

class DQN_PIL(nn.Module):
  def __init__(self):
    super(DQN_PIL, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=4, padding=0)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)  
    self.fc1 = nn.Linear(3136, 512)
    self.fc2 = nn.Linear(512,9)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(x.size(0),-1)
    x = F.relu(self.fc1(x))

    x = self.fc2(x)
    return x