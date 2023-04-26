#####################################################################
## CLASS THAT ALLOWS US TO MANAGE CSV DATASET IN PYTORCH FRAMEWORK ##
#####################################################################

#IMPORT OF THE REQUIRED LIBRARIES
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms 

class CSV_dataset(Dataset):
    def __init__(self, file_path, transform=None):
        # data loading
        xy = np.loadtxt(file_path, delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]
        
        self.x = xy[:,[0]]
        self.y = xy[:,[1]] #n_samples,1
        

        self.transform = transform
        
    
    def __getitem__(self, index):
        #dataset[0]
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample

    
    def __len__(self):
        #len(dataset)
        return self.n_samples

#CUSTOM TRANSFORM THAT TOOKS A NUMPY ARRAY AND TRASFORM IT IN A PYTORCH TENSOR
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)



