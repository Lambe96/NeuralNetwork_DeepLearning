######################################################################################################################################################
## THIS SCRIPT WILL PERFORM:                                                                                                                        ##
## 1) THE OPTIMIZATION OF THE HYPERPARAMETER AND OF THE ARCHITECTURE OF THE NETWORK OVER THE TRAINING SET (SPLITTED IN TRAINING AND VALIDATION SET).##
## 2) K-FOLD CROSS VALIDATION TRAINING PROCEDURE OVER THE TRAINING SET (SPLITTED IN TRAINING AND VALIDATION SET) WITH THE OPTIMIZED HYPERPARAMETER. ##
## 3) EVALUATION OF THE RESULTS OVER THE TEST DATASET.                                                                                              ##
## 4) PLOTS OF ALL (I HOPE!) THE RELEVANT INFORMATIONS ABOUT THE MODEL, THE OPTIMIZATION, THE TRAINING AND THE EVALUATION OF THE RESULTS.           ##
######################################################################################################################################################

#IMPORT OF THE REQUIRED LIBRARIES 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio
from DATASET import CSV_dataset, ToTensor
from torch.utils.data import SubsetRandomSampler
import torch.utils.data
from tqdm import tqdm as tqdm
from Hype_Select import Hype_Select
from sklearn.model_selection import KFold
import random

#CHOOSE GPU IF IT'S AVAILABLE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#REPRODUCIBILITY
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
if device=='cpu':
    torch.use_deterministic_algorithms(True)

#DATA  
train_data_path ='/Users/lambe/università/Laurea magistrale/secondo anno/Neural network and deep learning/homeworks/Homework_1-20210328/regression_task/regression_dataset/train_data.csv'
test_data_path = '/Users/lambe/università/Laurea magistrale/secondo anno/Neural network and deep learning/homeworks/Homework_1-20210328/regression_task/regression_dataset/test_data.csv'
data_to_study = CSV_dataset(train_data_path, transform=ToTensor())
#DATA USED TO OPTIMIZE THE HYPERPARAMETER 
data_to_study_train = torch.utils.data.Subset(data_to_study,torch.arange(int(len(data_to_study)*0.8)))
data_to_study_test = torch.utils.data.Subset(data_to_study,torch.arange(int(len(data_to_study)*0.8), len(data_to_study)))

# DEFINE THE OBJECT OF THE CLASS HYPE_SELECT THAT PERFORM THE OPTIMIZATION
model = Hype_Select(train_data=data_to_study_train,test_data=data_to_study_test)

#ATTRIBUTES OF THE CLASS THAT CORRESPONDS TO HYPERPARAMETERS TO BE OPTIMIZED
model.n_layers_range = [1,3]
model.n_units_range = [128,256]
model.batch_size_range = [4,50]
model.n_epochs_range = [100,1000]
model.learning_rate_range = [0.0001,0.01]
model.weight_decay_range = [1e-7, 0.1]
model.momentum_values = [0. , 0.9]
model.optimizer = ['Adam','SGD','RMSprop']

#START THE OPTIMIZATION THROUGH THE Hype_sel() METHOD
model.Hype_sel('Study_01',500,60000)

#BUILD THE OPTIMAL ARCHITECTURE FOR THE NETWORK IDENTIFIED BY THE OPTIMIZATION PROCEDURE
#SET THE ATTRIBUTES TO THE OPTIMIZED VALUES
model.build_opt_model()

# SET THE OPTIMAL HYPERPARAMETERS FOUND
num_epochs = model.n_epochs_range
batch_size = model.batch_size_range
learning_rate = model.learning_rate_range
weight_decay = model.weight_decay_range
momentum = model.momentum_values

# SET THE NUMBER OF FOLDS OF THE K-FOLD CROSS VALIDATION PROCEDURE
k = 5
splits=KFold(n_splits=k,shuffle=True,random_state=42)

#DATA TO TRAIN THE MODEL, WE SPLIT THEM IN 5 FOLDS
train_data = CSV_dataset(train_data_path, transform=ToTensor())

#DATA TO TEST THE MODEL AT THE END OF THE TRAINING
test_data = CSV_dataset(test_data_path, transform=ToTensor())

#SET THE LOSS FUNCTION USED IN THE OPTIMIZATION PROCEDURE
criterion = getattr(nn,model.loss_function)()

#USEFULL QUANTITIES TO VISUALIZE THE LEARNING PROCESS
sp = torch.from_numpy(np.linspace(-5,5,1000).astype(np.float32))
sp = sp.view(sp.shape[0],1)
fig, ax = plt.subplots(figsize=(12,7))
my_images = []

#LOG FOR THE MODELS TRAINED DURING THE CV PROCEDURE 
nets = []

#LOG FOR THE FOLDS PERFORMANCES
foldperf={}

#TRAINING LOOP WITH K-FOLD CROSS VALIDATION
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_data)))):
    #RESET THE NETWORK FOR EACH FOLD
    model.build_opt_model()

    net = model.model_arch

    optimizer = getattr(optim, model.optimizer)(net.parameters(), 
                    lr=learning_rate, weight_decay=weight_decay)
    if model.optimizer=='SGD':
        optimizer = getattr(optim, model.optimizer)(net.parameters(),
                    lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    net.to(device)

    #PERFORM THE SPLITING AND THE SAMPLING OF THE DATA
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(train_data, batch_size=batch_size, sampler=test_sampler)

    #LOG FOR THE MODEL PERFORMANCE
    history = {'train_loss': [], 'val_loss': []}

    #TRAINING LOOP
    for epoch in range(num_epochs):
        #MANAGE THE NETWORK IN THE LAST FOLD WHERE A GIF FILE OF THE LEARNING PROCESS IS PRODUCED
        if fold == (k-1):
            net.to(device)

        net.train()
        train_loss = []
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            #FORWARD PASS
            pred = net(x)
            loss = criterion(pred, y)

            #BACKWARD PASS
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 

            #SAVE LOSS VALUES FOR THE CURRENT BATCH
            loss_batch = loss.detach().cpu().numpy()
            train_loss.append(loss_batch) 

        # SAVE THE TRAIN LOSS
        train_loss = np.mean(train_loss)
        history['train_loss'].append(train_loss)

        #VALIDATION LOOP
        val_loss= []
        net.eval() 
        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                x = x.to(device)
                y = y.to(device)

                pred = net(x)
                loss = criterion(pred, y)

                loss_batch = loss.detach().cpu().numpy()
                val_loss.append(loss_batch)

            val_loss = np.mean(val_loss)
            history['val_loss'].append(val_loss) 
            
            #PRINT SOME INFO ABOUT THE LEARNING PROCESS
            print(f"Fold: {fold+1} Epoch:{epoch+1}/{num_epochs} AVG Training Loss:{train_loss} AVG Validation Loss:{val_loss}")


            #CREATE THE FRAMES FOR THE GIF FILE
            if fold == (k-1):
                if epoch % 5 == 0:
                    sp = sp.to('cpu')
                    net.to('cpu')
                    sp_pred = net(sp)
                    plt.cla()
                    ax.set_title('Regression Analysis', fontsize=35)
                    ax.set_xlabel('Independent variable', fontsize=24)
                    ax.set_ylabel('Dependent variable', fontsize=24)
                    ax.set_xlim(-4, 4)
                    ax.set_ylim(-4, 6)
                    ax.plot(train_data.x, train_data.y,'+', label='train data')
                    ax.plot(test_data.x, test_data.y,'*', label='test data')
                    ax.plot(sp.data.numpy(), sp_pred.data.numpy(), '-g', lw=2)
                    ax.text(1.0, 1.1, 'epoch = %d' % epoch, fontdict={'size': 20, 'color':  'red'})
                    ax.text(1.0, -1.1, 'fold = %d' % fold, fontdict={'size': 20, 'color':  'red'})
                    ax.text(1.0, 0, 'Loss = %.4f' % loss.data.cpu().numpy(),
                            fontdict={'size': 20, 'color':  'red'})

                    fig.canvas.draw()    
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    my_images.append(image)

    #SAVE THE MODEL PERFORMANCE
    foldperf['fold{}'.format(fold+1)] = history  
    #SAVE THE MODEL
    nets.append(net)

# PRODUCE THE GIF FILE THAT SHOWS THE LEARNING PROCESS
imageio.mimsave('./curve_1.gif', my_images, fps=10)

#COMPUTING THE PERFORMANCE OF EACH OF THE K NETWORKS OVER THE TEST DATASET TO CHOOSE THE BEST ONE AND HAVE A LOOK OVER THE VARIANCE OF THE PERFORMANCES
test_loss_nets = []
nets_names = []
with torch.no_grad():
    for ll in range(len(nets)):
        nets[ll].eval()
        nets_names.append(f'Net_{ll+1}')
        test_loss=[]
        for ii in range(len(test_data)):
            x = test_data[ii][0].to(device)
            y = test_data[ii][1].to(device)
            pred = nets[ll](x)
            loss = criterion(pred, y)
            loss.detach().cpu().numpy()
            test_loss.append(loss)
        test_loss_nets.append(np.mean(test_loss))

#PLOT THE TEST LOSSES OF THE MODELS
plt.figure(figsize=(12,8))
plt.bar(nets_names, test_loss_nets)
plt.xlabel('models')
plt.ylabel('test_losses')
plt.grid()
plt.savefig('models_performances.pdf')

#SELECT THE BEST NETWORK
net = nets[test_loss_nets.index(min(test_loss_nets))]

#COMPUTE THE PREDICTED VALUE AS THE AVERAGE OF THE PREDICTIONS OF THE DIFFERENT MODELS
pred = np.zeros((1000,1000))
for network in nets:
    pred += network(sp).data.numpy()
pred = pred/k
    
#PLOT THE AVERAGE PREDICTION
plt.figure(figsize=(12,8))
plt.plot(sp.data.numpy(), pred)
plt.plot(train_data.x, train_data.y,'+', label='train data')
plt.plot(test_data.x, test_data.y,'*', label='test data')
plt.grid()
plt.legend()
plt.savefig('data_mean.pdf')

#PLOT THE PREDICTION OF THE BEST PERFORMING NETWORK
plt.figure(figsize=(12,8))
plt.plot(sp.data.numpy(),net(sp).data.numpy())
plt.plot(train_data.x, train_data.y,'+', label='train data')
plt.plot(test_data.x, test_data.y,'*', label='test data')
plt.grid()
plt.legend()
plt.savefig('data_best.pdf')

#COMPUTE THE TRAIN AND VALIDATION LOSSES AS THE AVERAGE OVER THE K-FOLD
diz_ep = {'train_loss_ep':[],'val_loss_ep':[]}
for i in range(num_epochs):
      diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(k)]))
      diz_ep['val_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['val_loss'][i] for f in range(k)]))

#PLOT THE VALIDATION LOSSES FOR EACH FOLD
plt.figure(figsize=(12,8))
for f in range(k):
    plt.semilogy(foldperf['fold{}'.format(f+1)]['val_loss'],label=f'val_loss_fold_{f+1}')
plt.xlabel('Epoch')
plt.ylabel('val_loss')
plt.xscale('linear')
plt.yscale('linear')
plt.grid()
plt.legend()
plt.savefig('losses_for_each_fold.pdf')

#PLOT THE TRAIN AND VALIDATION LOSSES AS THE AVERAGE OVER THE K-FOLD
plt.figure(figsize=(12,8))
plt.semilogy(diz_ep['train_loss_ep'], label='Train loss')
plt.semilogy(diz_ep['val_loss_ep'], label='Validation loss')
plt.xscale('linear')
plt.yscale('linear')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.savefig('losses.pdf')

#PLOT THE WEIGHT HISTOGRAMS OF THE BEST PERFORMING NETWORK
if model.drop_out_range != 0. or isinstance(model.drop_out_range,list):
    step = 3
else:
    step = 2

fig, axs = plt.subplots(model.n_layers_range+1, 1, figsize=(12,8))

for ii in range(model.n_layers_range):
    if model.n_layers_range > 1:
        axs[ii].hist(net[ii*step].weight.data.cpu().numpy().flatten(), 50)
        axs[ii].set_title(f'layer {ii+1} of {model.n_layers_range} weights')
        [ax.grid() for ax in axs]
    else:
        axs[0].hist(net[ii*step].weight.data.cpu().numpy().flatten(), 50)
        axs[0].set_title(f'layer {ii+1} of {model.n_layers_range} weights')
axs[model.n_layers_range].hist(net[-1].weight.data.cpu().numpy().flatten(), 50)
axs[model.n_layers_range].set_title(f'output layer weights')
plt.tight_layout()
plt.savefig('weights.pdf')


#PLOT THE ACTIVATION PROFILES FOR DIFFERENT INPUTS OF EACH LAYER OF THE BEST PERFORMING NETWORK
inputs = [-3, -1, 0, 2, 4]
inputs_list = [ torch.tensor([x]).float() for x in inputs]


#A DICT TO STORE THE ACTIVATIONS
activation = {}
def getActivation(name):
  # the hook signature
  def hook(model, input, output):
    activation[name] = output.detach()
  return hook

for kk in range(len(inputs_list)):
    activation_layers = []
    # REGISTER FORWARD HOOKS ON THE ALL THE LAYERS
    for j in range(model.n_layers_range):
        activation_layers.append(net[2*j+1].register_forward_hook(getActivation(f'layer {j}')))  
    activation_layers.append(net[-1].register_forward_hook(getActivation(f'layer {model.n_layers_range}')))
    out = net(inputs_list[kk]) 

    for j in range(len(activation_layers)):
        activation_layers[j].remove()
    
    #GENERATE THE PLOTS
    fig, axs = plt.subplots(model.n_layers_range+1, 1, figsize=(12,8))
    for jj in range(model.n_layers_range):
        axs[jj].stem(activation[f'layer {jj}'].cpu().numpy(), use_line_collection=True)
        axs[jj].set_title(f'activation of hidden layer {jj+1} for x = {inputs[kk]}')
    axs[model.n_layers_range].stem(activation[f'layer {model.n_layers_range}'].cpu().numpy(), use_line_collection=True)
    axs[model.n_layers_range].set_title(f'activation output layer for x = {inputs[kk]}')
    plt.tight_layout()
    plt.savefig(f'activations_x_{inputs[kk]}.pdf')