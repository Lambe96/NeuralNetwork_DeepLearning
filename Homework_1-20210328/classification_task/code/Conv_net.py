########################################################################
##THIS CLASS DEFINE THE CNN MODEL.                                    ##
##THE ARCHITECTURE OF THE NETWORK IS BASED ON THAT OF LeNet-5 NETWORK.##
########################################################################

#IMPORT OF THE REQUIRED LIBRARIES
import torch.nn as nn
from torch.nn import functional as F

#INSTANTIATE THE CLASS
class LeNet_5(nn.Module):
    def __init__(self,drop_1=0.5,drop_2=0.3):
        super(LeNet_5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.drop1=nn.Dropout(p=drop_1)   

        self.fc1 = nn.Linear(16* 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.drop2=nn.Dropout(p=drop_2)
        self.fc3 = nn.Linear(84, 10)
    
    # FORWARD PASS METHOD
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x),kernel_size = 2, stride=2))

        x = F.relu(F.max_pool2d(self.conv2(x),kernel_size = 2, stride=2))
        x = self.drop1(x)

        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x