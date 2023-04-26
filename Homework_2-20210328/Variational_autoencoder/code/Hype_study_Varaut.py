#######################################################################################################################################################
## THIS SCRIPT WILL PERFORM:                                                                                                                         ##
## 1) THE OPTIMIZATION OF THE HYPERPARAMETER  OF THE NETWORKS OVER A SMALL PART OF THE TRAINING SET (SPLITTED IN TRAINING AND VALIDATION SET)        ##
## 2) K-FOLD CROSS VALIDATION TRAINING PROCEDURE OVER THE TRAINING SET (SPLITTED IN TRAINING AND VALIDATION SET) WITH THE OPTIMIZED HYPERPARAMETER.  ##
## 3) EVALUATION OF THE RESULTS OVER THE TEST DATASET.                                                                                               ##
## 4) PLOTS OF ALL (I HOPE!) THE RELEVANT INFORMATIONS ABOUT THE MODEL, THE OPTIMIZATION, THE TRAINING AND THE EVALUATION OF THE RESULTS.            ##
#######################################################################################################################################################

#IMPORT OF THE REQUIRED LIBRARIES 
from Hype_Select_Varaut import Hype_Select_Varaut
from Var_autoenc import Encoder, Decoder,Classifier, train_epoch, test_epoch
from torch.utils.data import  DataLoader,SubsetRandomSampler
import torch.utils.data
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

#LABELS
label_names=['t-shirt','trouser','pullover','dress','coat','sandal','shirt',
             'sneaker','bag','boot']

#REPRODUCIBILITY
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

# DEFINE THE OBJECT OF THE CLASS Hype_Select_Varaut THAT PERFORM THE OPTIMIZATION
model = Hype_Select_Varaut(train_data = data_to_study_train, test_data = data_to_study_test)

#ATTRIBUTES OF THE CLASS THAT CORRESPONDS TO HYPERPARAMETERS TO BE OPTIMIZED
model.encoded_space_dim = [2,10]
model.n_epochs_range = [10,150]
model.batch_size_range = [64,256]
model.learning_rate_range = [0.00001, 0.001]
model.weight_decay_range = [1e-6,1e-4]
model.optimizer = ['Adam','SGD','RMSprop']

#START THE OPTIMIZATION THROUGH THE Hype_sel() METHOD
model.Hype_sel('study_Varaut',150,43000)

encoded_space_dim = model.encoded_space_dim
num_epochs = model.n_epochs_range
batch_size = model.batch_size_range
learning_rate = model.learning_rate_range
weight_decay = model.weight_decay_range
momentum = model.momentum_range

# SET THE NUMBER OF FOLDS OF THE K-FOLD CROSS VALIDATION PROCEDURE
k = 10
splits = KFold(n_splits=k, shuffle=True, random_state=42 )

#LOG FOR THE MODELS TRAINED DURING THE CV PROCEDURE 
nets = []

#LOG FOR THE FOLDS PERFORMANCES
foldperf={}

#LOG TO TRACK THE PROGRESS OVER A FIXED SAMPLE DURING THE TRAINING
progress_img = []

#TRAINING LOOP WITH K-FOLD CROSS VALIDATION
for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_dataset)))):

    #PERFORM THE SPLITING AND THE SAMPLING OF THE DATA
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1)
    test_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler,num_workers=1)

    #CHOOSE GPU IF IT'S AVAILABLE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #RESET THE NETWORK FOR EACH FOLD
    encoder = Encoder(encoded_space_dim=encoded_space_dim)
    decoder = Decoder(encoded_space_dim=encoded_space_dim)

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
        ]

    optimizer = getattr(optim, model.optimizer)(params_to_optimize, 
                    lr=learning_rate, weight_decay=weight_decay)
    if model.optimizer=='SGD':
        optimizer = getattr(optim, model.optimizer)(params_to_optimize,
                    lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    #LOG FOR THE MODEL PERFORMANCE
    history = {'train_loss': [], 'test_loss': []}

    #TRAINING LOOP  
    for epoch in range(num_epochs):
        train_loss = train_epoch(encoder=encoder,
                                decoder=decoder,
                                device=device,
                                dataloader=train_loader,
                                optimizer=optimizer)

        test_loss = test_epoch(encoder=encoder,
                                decoder=decoder,
                                device=device,
                                dataloader=test_loader)

        
        #Adjust the measure of the train_loss and test_loss made inside the train_epoch and test_epoch functions
        #for the kfold process: in the functions we divide the total loss for len(dataloader.dataset)
        #but the dataset contains train_dataset+test_dataset sample. Instead we exploit only (k-1)/k part of
        #the dataset for training and 1/k for testing. Therefore we need a correction:
        #train_loss = train_loss*k/(k-1)
        #test_loss = test_loss*k

        #SAVE THE TRAIN AND THE VALIDATION LOSSES
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)

        #PRINT SOME INFO ABOUT THE LEARNING PROCESS     
        print(f"Fold:{fold+1} Epoch:{epoch+1}/{num_epochs} AVG Training Loss:{train_loss} AVG Test Loss:{test_loss}")

        #DURING LAST FOLD SAVE A PROCESSED SAMPLE
        if fold == k-1:
            img = test_dataset[0][0].unsqueeze(0).to(device)
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                progress_img.append(decoder(encoder(img)))
    #PLOT THE PROGRESS DURING LAST FOLD
    if fold == k-1:
        fig = plt.figure(figsize=(10,3*num_epochs))
        fig.suptitle('Progress')
        plt.axis('off')
        # CREATE num_epochs x 1 SUBFIGURES
        subfigs = fig.subfigures(nrows=num_epochs, ncols=1,)
        for row, subfig in enumerate(subfigs):
            subfig.suptitle(f'Epoch: {row+1}/{num_epochs}')
    
            # CREATE 1 x 2 SUBPLOTS 
            axs = subfig.subplots(nrows=1, ncols=2)
            axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[0].set_title('Original image')
            plt.axis('off')
            axs[1].imshow(progress_img[row].cpu().squeeze().numpy(), cmap='gist_gray')
            axs[1].set_title('Reconstructed image')
            plt.axis('off')
                
        fig.savefig('progress.pdf')

    #SAVE THE MODEL PERFORMANCE
    foldperf['fold{}'.format(fold+1)] = history
    
    #SAVE THE MODEL
    nets.append([encoder,decoder])


#COMPUTE THE AVERAGE TRAINING AND VALIDATION LOSS ACROSS THE K FOLDS
diz_ep = {'train_loss_ep':[],'test_loss_ep':[]}

for i in range(num_epochs):
    diz_ep['train_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['train_loss'][i] for f in range(k)]))
    diz_ep['test_loss_ep'].append(np.mean([foldperf['fold{}'.format(f+1)]['test_loss'][i] for f in range(k)]))

#PLOT AVERAGE LOSSES
plt.figure(figsize=(12,8))
plt.semilogy(diz_ep['train_loss_ep'], label='Train loss')
plt.semilogy(diz_ep['test_loss_ep'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.savefig('losses.pdf')

#PLOT THE VALIDATION LOSSES FOR EACH FOLD
plt.figure(figsize=(12,8))
for f in range(k):
    plt.semilogy(foldperf['fold{}'.format(f+1)]['test_loss'],label=f'val_loss_fold_{f+1}')
plt.xlabel('Epoch')
plt.ylabel('val_loss')
plt.xscale('linear')
plt.yscale('linear')
plt.grid()
plt.legend()
plt.savefig('losses_for_each_fold.pdf')

#SELECT THE BEST NETWORK
min_val_losses = [min(foldperf['fold{}'.format(f+1)]['test_loss']) for f in range(k)]
net = nets[min_val_losses.index(min(min_val_losses))]
encoder = net[0]
decoder = net[1]

#GET THE ENCODED REPRESENTATION OF THE TEST SAMPLES
encoded_samples = []
for sample in test_dataset:
    img = sample[0].unsqueeze(0).to(device)
    label = sample[1]
    # Encode image
    encoder.eval()
    with torch.no_grad():
        encoded_img  = encoder(img)
    # Append to list
    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"Encoded Var. {i}": enc for i, enc in enumerate(encoded_img)}
    encoded_sample['label'] = label
    encoded_samples.append(encoded_sample)

encoded_samples = pd.DataFrame(encoded_samples)
if model.encoded_space_dim > 2:
    #APPLY PCA IF THE LATENT SPACE HAS A DIMENSIONALITY >2
    pca = PCA(n_components=2)
    encoded_samples_reduced_PCA = pd.DataFrame(pca.fit_transform(encoded_samples.drop(['label'],axis=1)),columns=['Encoded Var. 1','Encoded Var. 2'])
    encoded_samples_reduced_PCA = encoded_samples_reduced_PCA.join(encoded_samples['label'])
    fig = px.scatter(encoded_samples_reduced_PCA, x='Encoded Var. 1', y='Encoded Var. 2', color=[label_names[l] for l in encoded_samples_reduced_PCA['label']], opacity=0.7)
    fig.write_image("Encoded_space_PCA.pdf") 

    #APPLY TSNE IF THE LATENT SPACE HAS A DIMENSIONALITY >2
    tsne = TSNE(n_components=2)
    encoded_samples_reduced_TSNE = pd.DataFrame(tsne.fit_transform(encoded_samples.drop(['label'],axis=1)),columns=['Encoded Var. 1','Encoded Var. 2'])
    encoded_samples_reduced_TSNE = encoded_samples_reduced_TSNE.join(encoded_samples['label'])
    fig = px.scatter(encoded_samples_reduced_TSNE, x='Encoded Var. 1', y='Encoded Var. 2', color=[label_names[l] for l in encoded_samples_reduced_TSNE['label']], opacity=0.7)
    fig.write_image("Encoded_space_TSNE.pdf") 
else:
    #SCATTER PLOT OF THE ELEMENT OF THE TEST DATASET IN THE ENCODED SPACE IF IT HAS A DIMENSIONALITY = 2
    fig = px.scatter(encoded_samples)
    px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1', color=[label_names[l] for l in encoded_samples['label']], opacity=0.7)
    fig.write_image('Scatter_encoded_space.pdf')

#GENERATE AND PLOT NEW SAMPLES FROM THE LATENT SPACE
encoder.eval()
decoder.eval()

with torch.no_grad():
    #MEAN AND STD OF TESTSET IN THE LATENT SPACE
    images, labels = iter(test_loader).next()
    images = images.to(device)
    latent = encoder(images)
    latent = latent.cpu()

    mean = latent.mean(dim=0)
    std = (latent - mean).pow(2).mean(dim=0).sqrt()

    #SAMPLE LATENT SPACE POINTS FROM A NORMAL DISTRIBUTION WITH MEAN AND STD COMPUTED ABOVE
    latent = torch.randn(100, model.encoded_space_dim)*std + mean

    #DECODE THE POINTS
    latent = latent.to(device)
    img_recon = decoder(latent)
    img_recon = img_recon.cpu()

    #PLOT THE GENERATED SAMPLE IN A GRID
    fig, ax = plt.subplots(figsize=(12, 8))
    npimg = torchvision.utils.make_grid(img_recon[:100],10,5).numpy()
    ax.imshow(np.transpose(npimg,(1,2,0)))
    fig.savefig('generated_samples_grid.pdf')

#FINE TUNING
classifier = Classifier(encoded_space_dim = model.encoded_space_dim)
criterion = nn.CrossEntropyLoss()
lr = 1e-4
l2 = 1e-4
batch_size = 128
optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay = weight_decay)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
num_epochs = 50
history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}

#TRAIN THE NETWORK
for epoch in range(num_epochs):
    train_loss,train_correct=0.0,0
    classifier.train()
    encoder.eval()
    for i ,(images, labels) in enumerate(train_loader):
        classifier.to(device)
        encoder.to(device)
        images,labels = images.to(device),labels.to(device)
        optimizer.zero_grad()
        Input = encoder(images)
        output = classifier(Input)
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

    classifier.eval()
    test_loss, test_correct= 0.0,0
    with torch.no_grad():
        for i,(images, labels) in enumerate(test_loader):
            classifier.to(device)
            encoder.to(device)
            images,labels = images.to(device),labels.to(device)
            Input = encoder(images)
            output = classifier(Input)
            loss=criterion(output,labels)
            test_loss+=loss.item()*images.size(0)
            scores, predictions = torch.max(output.data,1)
            test_correct+=(predictions == labels).sum().item()

    test_loss = test_loss / len(test_loader.sampler)
    test_acc = test_correct / len(test_loader.sampler) * 100

    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)

    print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".
    format(epoch + 1,num_epochs,train_loss,test_loss,train_acc,test_acc))



true_labels = []
preds = []
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

test_loss, test_correct= 0.0,0
for i,(images, labels) in enumerate(test_loader):
    images,labels = images.to(device),labels.to(device)
    classifier.to(device)
    encoder.to(device)

    Input = encoder(images)
    output = classifier(Input)
    loss=criterion(output,labels)
    #SAVE LOSS AND ACCURACY VALUES FOR THE CURRENT BATCH  
    test_loss+=loss.item()*images.size(0)
    scores, predictions = torch.max(output.data,1)
    test_correct+=(predictions == labels).sum().item()  
    preds.extend((torch.max(torch.exp(output), 1)[1]).data.cpu().numpy())
    true_labels.extend(labels.data.cpu().numpy())
#SAVE THE VALIDATION LOSS AND THE VALIDATION ACCURACY
test_loss = test_loss / len(test_loader.sampler)
test_acc = test_correct / len(test_loader.sampler)

os.chdir('..')
dir = os.path.join(os.getcwd(),'Fine_tuning')
if not os.path.exists(dir):
    os.mkdir(dir)
os.chdir(dir)

#COMPUTE THE CONFUSION MATRIX OF THE MODEL OVER THE TEST SET
cf_matrix = confusion_matrix(true_labels, preds)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in label_names],
                     columns = [i for i in label_names])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('confusion_matrix_fine_tuning.pdf')

# PLOT CLASSIFICATION TRAIN AND VALIDATION LOSSES
plt.figure(figsize=(12,8))
plt.semilogy(history['train_loss'], label='Train loss')
plt.semilogy(history['test_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.savefig('losses.pdf')

#PLOT CLASSIFICATION TRAIN AND VALIDATION ACCURACIES 
plt.figure(figsize=(12,8))
plt.semilogy(history['train_acc'], label='Train acc')
plt.semilogy(history['test_acc'], label='Validation acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.legend()
plt.savefig('Accuracies.pdf')