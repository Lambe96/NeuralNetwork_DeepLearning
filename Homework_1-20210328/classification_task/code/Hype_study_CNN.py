from Hype_Select_CNN import Hype_Select_CNN
from torch.utils.data import  DataLoader,SubsetRandomSampler, ConcatDataset
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


torch.manual_seed(42)

train_dataset = torchvision.datasets.FashionMNIST('classifier_data', train=True, download=True)
test_dataset  = torchvision.datasets.FashionMNIST('classifier_data', train=False, download=True)
data_to_study_train = torch.utils.data.Subset(train_dataset,torch.arange(10000))
data_to_study_test = torch.utils.data.Subset(train_dataset,torch.arange(10000,15000))

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])


train_dataset.transform = transform
test_dataset.transform = transform
data_to_study_train.transform = transform
data_to_study_test.transform = transform
model = Hype_Select_CNN(train_data = data_to_study_train, test_data = data_to_study_test)
model.n_epochs_range = [10,50]
model.batch_size_range = [64,256]
model.learning_rate_range = [0.0001,0.01]
model.weight_decay_range = [0.0001,0.01]
model.drop_out = [0.1,0.5]
model.momentum_values = [0.,0.9]
model.Hype_sel('study_CNN',100,43000)

num_epochs = model.n_epochs_range
batch_size = model.batch_size_range
learning_rate = model.learning_rate_range
weight_decay = model.weight_decay_range
momentum = model.momentum_values
drop_out = model.drop_out
dataset = ConcatDataset([train_dataset, test_dataset])

k=10
splits=KFold(n_splits=k,shuffle=True,random_state=42)
foldperf={}

#Initialize the loss and the optimizer

for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = LeNet_5(drop_1=drop_out[0],drop_2=drop_out[1]).to(device)
    optimizer = getattr(optim, model.optimizer)(net.parameters(), 
                    lr=learning_rate, weight_decay=weight_decay)
    if model.optimizer=='SGD':
        optimizer = getattr(optim, model.optimizer)(net.parameters(),
                    lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    criterion = getattr(nn,model.loss_function)()

    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

    for epoch in range(num_epochs):
        train_loss,train_correct=0.0,0
        model.train()
        for i ,(images, labels) in enumerate(train_loader):

            images,labels = images.to(device),labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output,labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            scores, predictions = torch.max(output.data, 1)
            train_correct += (predictions == labels).sum().item()

        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / len(train_loader.sampler) * 100

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        net.eval()
        test_loss, test_correct= 0.0,0
        with torch.no_grad():
            for i,(images, labels) in enumerate(test_loader):
                images,labels = images.to(device),labels.to(device)
                output = net(images)
                loss=criterion(output,labels)
                test_loss+=loss.item()*images.size(0)
                scores, predictions = torch.max(output.data,1)
                test_correct+=(predictions == labels).sum().item()

        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        foldperf['fold{}'.format(fold+1)] = history

        print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".
        format(epoch + 1,num_epochs,train_loss,test_loss,train_acc,test_acc))

diz_ep = {'train_loss_ep':[],'test_loss_ep':[],'train_acc_ep':[],'test_acc_ep':[]}

for i in range(num_epochs):
      diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(k)]))
      diz_ep['test_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_loss'][i] for f in range(k)]))
      diz_ep['train_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_acc'][i] for f in range(k)]))
      diz_ep['test_acc_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_acc'][i] for f in range(k)]))

# Plot losses
plt.figure(figsize=(12,8))
plt.semilogy(diz_ep['train_loss_ep'], label='Train loss')
plt.semilogy(diz_ep['test_loss_ep'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.savefig('losses.pdf')

# Plot accuracies 
plt.figure(figsize=(12,8))
plt.semilogy(diz_ep['train_acc_ep'], label='Train acc')
plt.semilogy(diz_ep['test_acc_ep'], label='Validation acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.savefig('Accuracies.pdf')

fig, axs = plt.subplots(3, 1, figsize=(12,8))
axs[0].hist(net.fc1.weight.data.cpu().numpy().flatten(), 50)
axs[0].set_title(f'linear layer 1 weights')
axs[1].hist(net.fc2.weight.data.cpu().numpy().flatten(), 50)
axs[1].set_title(f'linear layer 2 weights')
axs[2].hist(net.fc2.weight.data.cpu().numpy().flatten(), 50)
axs[2].set_title(f'linear layer 3 weights')
[ax.grid() for ax in axs]
    
plt.tight_layout()
plt.savefig('weights.pdf')

model_weights = [] 
conv_layers = [] 
model_children = list(net.children())
# counter to keep count of the conv layers
counter = 0 
# append all the conv layers and their respective weights to the list
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


# visualize the first conv layer filters
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

for i in range(act.size(0)//3):
        for j in range(act.size(0)//2):
           ax[i,j].imshow(act[k].detach().cpu().numpy())
           k+=1    
           plt.savefig('feature_map_1layer.pdf')

net.conv2.register_forward_hook(get_activation('conv2'))
data = train_dataset[19][0]
data=data.to(device)
data.unsqueeze_(0)
output = net(data)
k=0
act = activation['conv2'].squeeze()

fig, axarr = plt.subplots(4,4,figsize=(12, 16))
for i in range(4):
        for j in range(4):
          axarr[i,j].imshow(act[k].detach().cpu().numpy())
          k+=1  
          plt.savefig('feature_map_2layer.pdf')

os.chdir('..')
dir = os.path.join(os.getcwd(),'Model')
if not os.path.exists(dir):
    os.mkdir(dir)
os.chdir(dir)
path = os.path.join(os.getcwd(), 'net_.pth')
torch.save(net,path)