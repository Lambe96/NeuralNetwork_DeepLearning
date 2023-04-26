##################################################################################
## IN THIS FILE WE FIND:                                                        ##
## -THE CLASSES THAT DEFINE THE VARIETIONAL ENCODER AND DECODER                 ##
## -THE CLASS THAT DEFINES THE CLASSIFIER USED FOR THE SUPERVISED LEARNING TASK ##
## -THE FUNCTIONS THAT IMPLEMENTS THE TRAINING AND THE VALIDATION STEPS         ##
##################################################################################

#IMPORT THE REQUIRED LIBRARIES
import torch.nn as nn
import torch 
import numpy as np
import torch.nn.functional as F

#INSTANTIATE THE CLASS THAT DEFINE THE VARIATIONAL ENCODER
class Encoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()

        #CONVOLUTIONAL PART
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, 
                               stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, 
                               stride=2, padding=1)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0)

        #LINEAR PART 
        self.linear1 = nn.Linear(3*3*32, 128)
        self.linear2 = nn.Linear(128, encoded_space_dim)
        self.linear3 = nn.Linear(128, encoded_space_dim)

        #NORMAL DISTRIBUTION WITH MEAN 0 AND STD 1
        self.N = torch.distributions.Normal(0, 1)
        
        #GET THE SAMPLING ON THE GPU
        self.N.loc = self.N.loc.cuda() 
        self.N.scale = self.N.scale.cuda()

        #KULBACK-LEIBLER DIVERGENCE 
        self.kl = 0

    #FORWARD PASS
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #FLATTEN LAYER
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        #REPARAMETRIZATION TRICK TO ALLOW BACKPROPAGATION
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z

#INSTANTIATE THE CLASS THAT DEFINE THE VARIATIONAL DECODER
class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()

        #LINEAR PART
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True)
        )

        #UNFLATTEN LAYER
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        #TRANSPOSED CONVOLUTIONAL PART
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        
    #FORWARD PASS
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

#INSTANTIATE THE CLASS THAT DEFINES THE CLASSIFIER USED FOR FINE TUNING 
class Classifier(nn.Module):
    
    def __init__(self, encoded_space_dim, drop_1=0.2,drop_2=0.1):
        super().__init__()

        #NETWORK ARCHITECTURE: JUST ONE HIDDEN LAYER
        self.input = nn.Linear(in_features=encoded_space_dim, out_features=120)
        self.hidden = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)
        
        #ACTIVATION FUNCTION
        self.act = nn.ReLU()

        #DROPOUT
        self.drop_1 = nn.Dropout(p = drop_1)
        self.drop_2 = nn.Dropout(p = drop_2)

    #FORWARD PASS
    def forward(self, x):
        x = self.act(self.input(x))
        x = self.drop_1(x)
        x = self.act(self.hidden(x))
        x = self.drop_2(x)
        x = self.out(x)

        return x

#TRAINING FUNCTION
def train_epoch(encoder, decoder, device, dataloader, optimizer):
    train_loss = 0.0
    encoder.to(device)
    decoder.to(device)
    encoder.train()
    decoder.train()
    #ITERATE THE DATALOADER IGNORING THE LABELS
    for image_batch, _ in dataloader: 
        image_batch = image_batch.to(device)
        #ENCODED IMAGE
        encoded_data = encoder(image_batch)
        #DECODED IMAGE
        decoded_data = decoder(encoded_data)
        #LOSS + KL DIVERGENCE
        loss = ((image_batch - decoded_data)**2).sum() + encoder.kl
        
        #OPTIMIZATION STEP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss/len(dataloader.sampler)


#VALIDATION/TEST FUNCTION
def test_epoch(encoder, decoder, device, dataloader):
    # Set evaluation mode for encoder and decoder
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    test_loss = 0.0
    with torch.no_grad():
        #ITERATE THE DATALOADER IGNORING THE LABELS
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            #ENCODED IMAGE
            encoded_data = encoder(image_batch)
            #DECODED IMAGE
            decoded_data = decoder(encoded_data)
            #LOSS+KL DIVERGENCE
            loss = ((image_batch - decoded_data)**2).sum() + encoder.kl
            test_loss += loss.item()
    return test_loss / len(dataloader.sampler)