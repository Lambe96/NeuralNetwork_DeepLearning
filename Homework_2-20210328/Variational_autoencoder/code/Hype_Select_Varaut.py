##########################################################################################################
## THIS CLASS IMPLEMENT THE HYPERPARAMETER OPTIMIZATION WITHIN THE OPTUNA FRAMEWORK.                    ##
## IT INSTANTIATE AN OBJECT WITH A SERIES OF ATTRIBUTES THAT CORRESPONDS TO THE HYPERPARAMETER OF THE   ##
## MODEL. ALL OF THEM EXCEPT FOR THE ACTIVATION FUNCTION CAN BE OPTIMIZED THROUGH                       ## 
## THE Hype_Sel() METHOD.                                                                               ##
## TO PERFORM THE OPTIMIZATION OVER AN HYPERPARAMETER WE SET IT TO A LIST CONTAINING THE INTERVAL       ##
## OVER WHICH WE WANT TO RUN THE OPTIMIZATION, OR A SET OF DESCRETE VALUES OR STRINGS.                  ##
##########################################################################################################

#IMPORT THE REQUIRED LIBRARIES
from numpy.core.numeric import NaN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from Var_autoenc import Encoder, Decoder
from Var_autoenc import train_epoch, test_epoch
import optuna
from optuna.trial import TrialState
import os 
from itertools import combinations as cmb
import inspect

#INSTANTIATE THE CLASS
class Hype_Select_Varaut(nn.Module):
    def __init__(self, 
                train_data, 
                test_data,
                n_epochs_range = 50,
                batch_size_range = 1, 
                learning_rate_range = 0.001,
                optimizer = 'Adam', 
                activation_func: str = 'ReLU',
                weight_decay_range = 0.,
                encoded_space_dim = 2,
                momentum_range = 0.):

        super(Hype_Select_Varaut, self).__init__()
        ####################################### INPUT HANDLING #################################
        if not (isinstance(n_epochs_range, int) or isinstance(n_epochs_range,list)):
            raise TypeError('"n_epochs_range" must be an integer or a list made by two integers')
        if isinstance(n_epochs_range,list) and (len(n_epochs_range) != 2):
            raise IndexError('"n_epochs_range" must be of lenght 2')

        if not (isinstance(batch_size_range, int) or isinstance(batch_size_range,list)):
            raise TypeError('"batch_size_rangee" must be an integer or a list made by two integers')
        if isinstance(batch_size_range,list) and (len(batch_size_range) != 2):
            raise IndexError('"batch_size_range" must be of lenght 2')

        if not (isinstance(learning_rate_range, float) or isinstance(learning_rate_range,list)):
            raise TypeError('"learning_rate_range" must be a float value or a list made by two float values')
        if isinstance(learning_rate_range,list) and (len(learning_rate_range) != 2):
            raise IndexError('"learning_rate_range" must be of lenght 2')

        if not (isinstance(weight_decay_range, float) or isinstance(weight_decay_range,list)):
            raise TypeError('"weight_decay_range" must be a float value or a list made by two float values')
        if isinstance(weight_decay_range,list) and (len(weight_decay_range) != 2):
            raise IndexError('"weight_decay_range" must be of lenght 2')
        if not (isinstance(momentum_range, float) or isinstance(momentum_range,list)):
            raise TypeError('"momentum_range" must be a float value or a list made by two float values')
        if isinstance(momentum_range,list) and (len(momentum_range) != 2):
            raise IndexError('"momentum_range" must be of lenght 2')
        
        if not (isinstance(optimizer, str) or isinstance(optimizer,list)):
            raise TypeError('"optimizer" must be of type string or list')
        
        if not isinstance(activation_func,str):
            raise TypeError('"activation_func" must be of type string')

        ################ OBJECT INITIALIZATION ################
        self.train_data = train_data
        self.test_data = test_data
        self.n_epochs_range = n_epochs_range
        self.batch_size_range = batch_size_range
        self.learning_rate_range = learning_rate_range
        self.optimizer = optimizer
        self.weight_decay_range = weight_decay_range
        self.momentum_range = momentum_range
        self.activation_func = activation_func
        self.optimized_param = None
        self.encoded_space_dim = encoded_space_dim

    def Hype_sel(self, study_name:str, n_trials, timeout, sampler=optuna.samplers.TPESampler(), direction='minimize'):
        
        #####################################################################
        ## THIS METHOD PERFORMS THE STUDY OF THE SELECTED HYPERPARAMETERS, ##
        ## AND GENERATE SOME USEFUL OUTPUTS.                               ##
        #####################################################################
        
        #DIRECTORY TO STORE THE STUDY RESULTS
        dir = os.path.join(os.getcwd(),study_name+'_results')
        if not os.path.exists(dir):
            os.mkdir(dir)
        os.chdir(dir)

        #FUNCTION TO BE OPTIMIZED
        def objective(trial):

            #BATCH SIZE            
            if isinstance(self.batch_size_range, int):
                batch_size = self.batch_size_range
            else:
                batch_size = trial.suggest_int('batch size',self.batch_size_range[0], self.batch_size_range[1])

            #DATALOADERS
            train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True, num_workers=1)
            test_loader =  DataLoader(dataset=self.test_data, batch_size=batch_size, shuffle=False, num_workers=1)

            #CHOOSE GPU IF IT'S AVAILABLE
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
            
            #NUMBER OF EPOCHS
            if isinstance(self.n_epochs_range,int):
                num_epochs = self.n_epochs_range
            else:
                num_epochs = trial.suggest_int('number of epochs', self.n_epochs_range[0], self.n_epochs_range[1])

            #LEARNING RATE
            if isinstance(self.learning_rate_range, float):
                lr = self.learning_rate_range
            else:
                lr = trial.suggest_float("learning rate", self.learning_rate_range[0],self.learning_rate_range[1], log=True)

            #L2 REGULARIZATION
            if isinstance(self.weight_decay_range, float):
                weight_decay = self.weight_decay_range
            else:
                weight_decay = trial.suggest_float('l2 regularization', self.weight_decay_range[0], self.weight_decay_range[1],log=True)

            #DIMENSIONALITY OF THE LATENT SPACE
            if isinstance(self.encoded_space_dim, int):
                encoded_space_dim = self.encoded_space_dim
            else:
                encoded_space_dim = trial.suggest_int('Encoded space dim.',self.encoded_space_dim[0], self.encoded_space_dim[1])

            #INSTANTIATE THE ENCODER AND THE DECODER 
            encoder = Encoder(encoded_space_dim=encoded_space_dim)
            decoder = Decoder(encoded_space_dim=encoded_space_dim)

            params_to_optimize = [{'params': encoder.parameters()},
                                    {'params': decoder.parameters()}
]
            #OPTIMIZER
            if isinstance(self.optimizer,list):
                opt = trial.suggest_categorical('optimizer',self.optimizer)
            else:
                opt = self.optimizer
                
            optimizer = getattr(optim, opt)(params_to_optimize, lr=lr, weight_decay=weight_decay)
            
            #MOMENTUM IF NEEDED
            if 'momentum' in inspect.getfullargspec(getattr(optim, opt))[0]:
                if isinstance(self.momentum_range, float):
                    momentum = self.momentum_range
                else:
                    momentum = trial.suggest_float('momentum', self.momentum_range[0],self.momentum_range[1])
                optimizer = getattr(optim, opt)(params_to_optimize, lr=lr, weight_decay=weight_decay, momentum=momentum)

            #TRAINING LOOP
            for epoch in range(num_epochs):
                train_loss = train_epoch(encoder=encoder,
                            decoder=decoder,
                            device=device,
                            dataloader=train_loader,
                            optimizer=optimizer)
                
                #VALIDATION PROCESS  
                test_loss = test_epoch(encoder=encoder,
                                       decoder=decoder,
                                       device=device,
                                       dataloader=test_loader)
                                
                
                trial.report(test_loss, epoch)

                #HANDLE PRUNING PROCESS IN BASE OF THE TEST_LOSS VALUES.
                #THE PRUNER IS THE MEDIAN PRUNER, DEFAULT FOR OPTUNA.
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return test_loss
        
        #CREATE AND RUN THE STUDY
        study = optuna.create_study(study_name=study_name, direction=direction, sampler=sampler, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        #SELECT THE BEST TRIAL
        trial = study.best_trial

        #WRITE A FILE WITH THE INFO ABOUT THE STUDY AND THE OPTIMIZED PARAMETERS
        f = open(study_name+'_bestparams.txt','w+')
        f.write("Study statistics: \n")
        f.write(f"  Number of finished trials:  {len(study.trials)}\n")
        f.write(f"  Number of pruned trials:  {len(pruned_trials)}\n")
        f.write(f"  Number of complete trials:  {len(complete_trials)}\n")
        f.write('Best trial:\n')
        f.write(f"  Value:  {trial.value}\n")
        f.write('   Params:\n')
        for key, value in trial.params.items():
            f.write("    {}: {}\n".format(key, value))
        f.close()

        #DIRECTORY TO STORE ALL THE VISUALIZATION
        dir = os.path.join(os.getcwd(),'Visualization')
        if not os.path.exists(dir):
            os.mkdir(dir)
        os.chdir(dir)

        #OPTUNA PLOTS
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("importance.pdf")
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("history.pdf")            
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())[:4]
        fig = optuna.visualization.plot_slice(study, params=params)
        fig.write_image(f'slice.pdf')
        if len(importance) >= 4:
            for i in range(4):
                params = list(cmb(importance,2))
                params = params[:4]
                for param in params:
                    fig = optuna.visualization.plot_contour(study,params=param)
                    fig.write_image(f"countur_{param[0]}_{param[1]}.pdf")
        else:
            for i in range(len(importance)):
                params = list(cmb(importance,2))
                for param in params:
                    fig = optuna.visualization.plot_contour(study,params=param)
                    fig.write_image(f"countur_{param[0]}_{param[1]}.pdf")
        



        #PRINTED OUTPUT
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
        print("Best trial:")
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        
        self.optimized_param = trial.params


        #SET THE ATTRIBUTES OF THE OBJECT TO THE ONE OF THE BEST TRIAL  
        if isinstance(self.optimizer,list):
            self.optimizer = trial.params['optimizer']

        if isinstance(self.batch_size_range,list):
            self.batch_size_range = self.optimized_param.get('batch size')

        if isinstance(self.n_epochs_range, list):
            self.n_epochs_range = self.optimized_param.get('number of epochs')

        if isinstance(self.learning_rate_range, list):
            self.learning_rate_range = self.optimized_param.get('learning rate')

        if isinstance(self.weight_decay_range, list):
            self.weight_decay_range = self.optimized_param.get('l2 regularization')

        if isinstance(self.momentum_range,list):
            self.momentum_range = self.optimized_param.get('momentum')

        if isinstance(self.encoded_space_dim, list):
            self.encoded_space_dim = self.optimized_param.get('Encoded space dim.')
        
        return                 

