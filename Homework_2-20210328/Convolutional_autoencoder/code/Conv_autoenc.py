##################################################################################
## IN THIS FILE WE FIND:                                                        ##
## -THE CLASSES THAT DEFINE THE CONVOLUTIONAL ENCODER AND DECODER               ##
## -THE CLASS THAT DEFINES THE CLASSIFIER USED FOR THE SUPERVISED LEARNING TASK ##
## -THE FUNCTIONS THAT IMPLEMENTS THE TRAINING AND THE VALIDATION STEPS         ##
##################################################################################

#IMPORT THE REQUIRED LIBRARIES
import torch.nn as nn
import torch 
import numpy as np

#INSTANTIATE THE CLASS THAT DEFINE THE CONVOLUTIONAL ENCODER
class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        #CONVOLUTIONAL PART
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, 
                      stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, 
                      stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, 
                      stride=2, padding=0),
            nn.ReLU(True)
        )
        
        #FLATTEN LAYER
        self.flatten = nn.Flatten()

        #LINEAR PART
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(in_features=288, out_features=64),
            nn.ReLU(True),
            # Second linear layer
            nn.Linear(in_features=64, out_features=encoded_space_dim)
        )
    #FORWARD PASS
    def forward(self, x):
        # APPLY CONVOLUIONS
        x = self.encoder_cnn(x)
        # FLAT THE NETWORK
        x = self.flatten(x)
        #APPLY THE LINEAR PART
        x = self.encoder_lin(x)
        return x

#INSTANTIATE THE CLASS THAT DEFINE THE CONVOLUTIONAL DECODER
class Decoder(nn.Module): 
    def __init__(self, encoded_space_dim):
        super().__init__()

        #LINEAR PART
        self.decoder_lin = nn.Sequential(
            nn.Linear(in_features=encoded_space_dim, out_features=64),
            nn.ReLU(True),
            nn.Linear(in_features=64, out_features=288),
            nn.ReLU(True)
        )

        #UNFLATTEN
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))

        #TRANSPOSED CONVOLUTIONAL PART
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, 
                               stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, 
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, 
                               stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        #APPLY LINEAR PART
        x = self.decoder_lin(x)
        #UNFLUTTEN THE NETWORK
        x = self.unflatten(x)
        #APPLY THE CONVOLUTIONAL PART
        x = self.decoder_conv(x)
        #APPLY A SIGMOID TO FORCE THE OUTPUT TO ASSUME VALID PIXEL VALUES
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


# TRAINING FUNCTION
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    train_loss = []
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
        #COMPUTING THE LOSS
        loss = loss_fn(decoded_data, image_batch)
        
        #OPTIMIZATION STEP
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #SAVE THE TRAIN LOSS
        train_loss.append(loss.detach().cpu().numpy())
    return np.mean(train_loss)


#VALIDATION/TEST FUNCTION
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): 
        #DEFINE THE LISTS TO STORE THE OUTPUTS FOR EACH BATCH
        conc_out = []
        conc_label = []
        # ITERATE THE DATALOADER IGNORING THE LABELS
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            #ENCODED IMAGE
            encoded_data = encoder(image_batch)
            #DECODED IMAGE
            decoded_data = decoder(encoded_data)
            # APPEND THE NETWORK OUTPUT AND THE ORIGINAL IMAGE TO THE LISTS
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # CREATE TWO TENSORS WITH THE IMAGES IN THE TWO LISTS
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # EVALUATE THE LOSS OF THE BATCH
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data

        

    
