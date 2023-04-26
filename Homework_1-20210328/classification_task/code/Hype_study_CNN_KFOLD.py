######################################################################################################################################################
## THIS SCRIPT WILL PERFORM:                                                                                                                        ##
## 1) THE OPTIMIZATION OF THE HYPERPARAMETER  OF THE NETWORK OVER A SMALL PART OF THE TRAINING SET (SPLITTED IN TRAINING AND VALIDATION SET)        ##
## 2) K-FOLD CROSS VALIDATION TRAINING PROCEDURE OVER THE TRAINING SET (SPLITTED IN TRAINING AND VALIDATION SET) WITH THE OPTIMIZED HYPERPARAMETER. ##
## 3) EVALUATION OF THE RESULTS OVER THE TEST DATASET.                                                                                              ##
## 4) PLOTS OF ALL (I HOPE!) THE RELEVANT INFORMATIONS ABOUT THE MODEL, THE OPTIMIZATION, THE TRAINING AND THE EVALUATION OF THE RESULTS.           ##
######################################################################################################################################################

#IMPORT OF THE REQUIRED LIBRARIES 
from Hype_Select_CNN import Hype_Select_CNN
from torch.utils.data import  DataLoader,SubsetRandomSampler
import torch.utils.data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from Conv_net import LeNet_5
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
import random
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

#REPRODUCIBILITY
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
torch.manual_seed(42)

#DATA
train_dataset = torchvision.datasets.FashionMNIST('classifier_data', train=True, download=True)
test_dataset  = torchvision.datasets.FashionMNIST('classifier_data', train=False, download=True)
#DATA USED TO OPTIMIZE THE HYPERPARAMETER 
data_to_study_train = torch.utils.data.Subset(train_dataset,torch.arange(10000))
data_to_study_test = torch.utils.data.Subset(train_dataset,torch.arange(10000,15000))

#PYTORCH TRANSFORMS
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])
train_dataset.transform = transform
test_dataset.transform = transform
data_to_study_train.transform = transform
data_to_study_test.transform = transform

# DEFINE THE OBJECT OF THE CLASS HYPE_SELECT_CNN THAT PERFORM THE OPTIMIZATION
model = Hype_Select_CNN(train_data = data_to_study_train, test_data = data_to_study_test)

#ATTRIBUTES OF THE CLASS THAT CORRESPONDS TO HYPERPARAMETERS TO BE OPTIMIZED
model.n_epochs_range = [10,100]
model.batch_size_range = [64,256]
model.learning_rate_range = [0.00001,0.01]
model.weight_decay_range = [0.00001,0.01]
model.drop_out = [0.1,0.5]
model.momentum_values = [0.,0.9]
model.optimizer = ['Adam', 'SGD', 'RMSprop']

#START THE OPTIMIZATION THROUGH THE Hype_sel() METHOD
model.Hype_sel('study_CNN',300,43000)

# SET THE OPTIMAL HYPERPARAMETERS FOUND
num_epochs = model.n_epochs_range
batch_size = model.batch_size_range
learning_rate = model.learning_rate_range
weight_decay = model.weight_decay_range
momentum = model.momentum_values
drop_out = model.drop_out

# SET THE NUMBER OF FOLDS OF THE K-FOLD CROSS VALIDATION PROCEDURE
k=10
splits=KFold(n_splits=k,shuffle=True,random_state=42)

#LOG FOR THE MODELS TRAINED DURING THE CV PROCEDURE 
nets = []

#LOG FOR THE FOLDS PERFORMANCES
foldperf={}

#TRAINING LOOP WITH K-FOLD CROSS VALIDATION
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):
    
    #PERFORM THE SPLITING AND THE SAMPLING OF THE DATA
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler)
    
    #CHOOSE GPU IF IT'S AVAILABLE
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #RESET THE NETWORK FOR EACH FOLD
    net = LeNet_5(drop_1=drop_out[0],drop_2=drop_out[1]).to(device)
    optimizer = getattr(optim, model.optimizer)(net.parameters(), 
                    lr=learning_rate, weight_decay=weight_decay)
    if model.optimizer == 'SGD':
        optimizer = getattr(optim, model.optimizer)(net.parameters(),
                    lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    #SET THE LOSS FUNCTION USED IN THE OPTIMIZATION PROCEDURE
    criterion = getattr(nn,model.loss_function)()

    #LOG FOR THE MODEL PERFORMANCE
    history = {'train_loss': [], 'validation_loss': [],'train_acc':[],'validation_acc':[]}

    #TRAINING LOOP
    for epoch in range(num_epochs):
        train_loss,train_correct=0.0,0
        net.train()

        for i ,(images, labels) in enumerate(train_loader):

            images,labels = images.to(device),labels.to(device)

            #FORWARD PASS
            output = net(images)
            loss = criterion(output,labels)

            #BACKWARD PASS
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #SAVE LOSS AND ACCURACY VALUES FOR THE CURRENT BATCH            
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()

        #SAVE THE TRAIN LOSS AND THE TRAIN ACCURACY
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        #VALIDATION LOOP
        net.eval()
        val_loss, val_correct= 0.0,0
        with torch.no_grad():
            for i,(images, labels) in enumerate(val_loader):
                images,labels = images.to(device),labels.to(device)

                output = net(images)
                loss=criterion(output,labels)

                #SAVE LOSS AND ACCURACY VALUES FOR THE CURRENT BATCH  
                val_loss+=loss.item()*images.size(0)
                scores, predictions = torch.max(output.data,1)
                val_correct+=(predictions == labels).sum().item()
        
        #SAVE THE VALIDATION LOSS AND THE VALIDATION ACCURACY
        val_loss = val_loss / len(val_loader.sampler)
        val_acc = val_correct / len(val_loader.sampler) * 100

        history['validation_loss'].append(val_loss)
        history['validation_acc'].append(val_acc)

        #PRINT SOME INFO ABOUT THE LEARNING PROCESS
        print("Fold: {} Epoch:{}/{} AVG Training Loss:{:.3f} AVG Validation Loss:{:.3f} AVG Training Acc {:.2f} % AVG Validation Acc {:.2f} %".
        format(fold+1,epoch + 1,num_epochs,train_loss,val_loss,train_acc,val_acc))

    #SAVE THE MODEL PERFORMANCE
    foldperf['fold{}'.format(fold+1)] = history
    #SAVE THE MODEL
    nets.append(net)

#COMPUTING THE PERFORMANCE OF EACH OF THE K NETWORKS OVER THE TEST DATASET TO CHOOSE THE BEST ONE AND HAVE A LOOK OVER THE VARIANCE OF THE PERFORMANCES
test_acc_nets = []
test_loss_nets = []
nets_names = []
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
nets_pred = {}
with torch.no_grad():
    for ii in range(len(nets)):
        net=nets[ii].to(device)
        net.eval()
        nets_names.append(f'Net_{ii+1}')
        test_loss, test_correct= 0.0,0
        true_labels = []
        preds = []

        for i,(images, labels) in enumerate(test_loader):
            images,labels = images.to(device),labels.to(device)
            output = net(images)
            loss=criterion(output,labels)
            #SAVE LOSS AND ACCURACY VALUES FOR THE CURRENT BATCH  
            test_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            test_correct+=(predictions == labels).sum().item()  
            print(test_correct)
            preds.extend((torch.max(torch.exp(output), 1)[1]).data.cpu().numpy())
            true_labels.extend(labels.data.cpu().numpy())
        nets_pred[nets_names[-1]] = preds
        
        #SAVE THE VALIDATION LOSS AND THE VALIDATION ACCURACY
        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler)
        test_loss_nets.append(test_loss)
        test_acc_nets.append(test_acc)

#PLOT THE TEST ACCURACIES OF THE MODELS
plt.subplots(figsize=(12,8))
plt.bar(nets_names, test_acc_nets)
plt.xlabel('models')
plt.ylabel('accuracy')
plt.grid()
plt.savefig('models_performances.pdf')

#SELECT THE BEST NETWORK
net = nets[test_loss_nets.index(min(test_loss_nets))].to(device)

#COMPUTE THE CONFUSION MATRIX OF THE BEST NETWORK
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

cf_matrix = confusion_matrix(true_labels, nets_pred[nets_names[test_loss_nets.index(min(test_loss_nets))]] )
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrix_best.pdf')

#COMPUTE THE CONFUSION MATRIX WITH THE PREDICTIONS FROM THE MAJORITY VOTING OF THE NETWORKS 
#(IF TWO LABELS HAVE THE SAME NUMBER OF VOTES, THEN WE TAKE THE PREDICTION OF THE BEST NETWORK)
df = pd.DataFrame.from_dict(nets_pred)
df = df.mode(axis=1)
unsure_indexes = list(df.loc[df[1].notnull(), :].index)
majority_pred = list(df[0])
for i in unsure_indexes:
  majority_pred[i] = nets_pred[nets_names[test_loss_nets.index(min(test_loss_nets))]][i]

cf_matrix = confusion_matrix(true_labels, majority_pred )
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrix_majority.pdf')
        
#COMPUTE THE TRAIN AND VALIDATION LOSSES AND ACCURACIES AS THE AVERAGE OVER THE K-FOLD
diz_ep = {'train_loss_ep':[],'validation_loss_ep':[],'train_acc_ep':[],'validation_acc_ep':[]}

for i in range(num_epochs):
      diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(k)]))
      diz_ep['validation_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['validation_loss'][i] for f in range(k)]))
      diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(k)]))
      diz_ep['validation_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['validation_acc'][i] for f in range(k)]))

#PLOT THE TRAIN AND VALIDATION LOSSES AS THE AVERAGE OVER THE K-FOLD
plt.figure(figsize=(12,8))
plt.semilogy(diz_ep['train_loss_ep'], label='Train loss')
plt.semilogy(diz_ep['validation_loss_ep'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.savefig('losses.pdf')

#PLOT THE TRAIN AND VALIDATION ACCURACIES AS THE AVERAGE OVER THE K-FOLD
plt.figure(figsize=(12,8))
plt.semilogy(diz_ep['train_acc_ep'], label='Train acc')
plt.semilogy(diz_ep['validation_acc_ep'], label='Validation acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.savefig('Accuracies.pdf')

#PLOT THE VALIDATION LOSSES FOR EACH FOLD
plt.figure(figsize=(12,8))
for f in range(k):
    plt.semilogy(foldperf['fold{}'.format(f+1)]['validation_loss'],label=f'val_loss_fold_{f+1}')
plt.xlabel('Epoch')
plt.ylabel('val_loss')
plt.xscale('linear')
plt.yscale('linear')
plt.grid()
plt.legend()
plt.savefig('losses_for_each_fold.pdf')

#PLOT THE WEIGHT HISTOGRAMS OF THE BEST PERFORMING NETWORK
fig, axs = plt.subplots(3, 1, figsize=(12,8))
axs[0].hist(net.fc1.weight.data.cpu().numpy().flatten(), 50)
axs[0].set_title(f'linear layer 1 weights')
axs[1].hist(net.fc2.weight.data.cpu().numpy().flatten(), 50)
axs[1].set_title(f'linear layer 2 weights')
axs[2].hist(net.fc3.weight.data.cpu().numpy().flatten(), 50)
axs[2].set_title(f'linear layer 3 weights')
[ax.grid() for ax in axs]
    
plt.tight_layout()
plt.savefig('weights.pdf')

#RETRIEVE THE FILTERS OF THE CONVOLUTIONAL LAYERS
model_weights = [] 
conv_layers = [] 
model_children = list(net.children())
counter = 0 

for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)


#PLOT THE CONVOLUTIONAL LAYERS FILTERS
plt.figure(figsize=(12, 8))
for i, filter in enumerate(model_weights[0]):
    plt.subplot(2, 3, i+1)
    plt.imshow(filter[0, :, :].detach().cpu().numpy(), cmap='cividis')
    plt.axis('off')
plt.savefig('filter_1convlayer.pdf')

plt.figure(figsize=(12, 8))
for i, filter in enumerate(model_weights[1]):
    plt.subplot(4, 4, i+1)
    plt.imshow(filter[0, :, :].detach().cpu().numpy(), cmap='cividis')
    plt.axis('off')
plt.savefig('filter_2convlayer.pdf')


#RETRIEVE THE FEATURE MAPS OF THE FIRST CONVOLUTIONAL LAYER
activation = {}
def get_activation(name):
    def hook(model, input, out):
        activation[name] = out.detach()
    return hook


net.conv1.register_forward_hook(get_activation('conv1'))
data = train_dataset[19][0]
data=data.to(device)
data.unsqueeze_(0)
output = net(data)
k=0
act = activation['conv1'].squeeze()
fig,ax = plt.subplots(2,3,figsize=(12, 15))

#PLOT THE FEATURE MAP OF THE FIRST LAYER 
for i in range(act.size(0)//3):
        for j in range(act.size(0)//2):
           ax[i,j].imshow(act[k].detach().cpu().numpy())
           k+=1    
           plt.savefig('feature_map_1layer.pdf')

#RETRIEVE THE FEATURE MAPS OF THE SECOND CONVOLUTIONAL LAYER
net.conv2.register_forward_hook(get_activation('conv2'))
data = train_dataset[19][0]
data=data.to(device)
data.unsqueeze_(0)
output = net(data)
k=0
act = activation['conv2'].squeeze()

#PLOT THE FEATURE MAPS OF THE SECOND LAYER 
fig, axarr = plt.subplots(4,4,figsize=(12, 16))
for i in range(4):
        for j in range(4):
          axarr[i,j].imshow(act[k].detach().cpu().numpy())
          k+=1  
          plt.savefig('feature_map_2layer.pdf')