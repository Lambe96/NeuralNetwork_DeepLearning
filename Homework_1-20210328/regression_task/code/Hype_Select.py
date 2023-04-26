##########################################################################################################
## THIS CLASS IMPLEMENT THE HYPERPARAMETER OPTIMIZATION WITHIN THE OPTUNA FRAMEWORK.                    ##
## IT INSTANTIATE AN OBJECT WITH A SERIES OF ATTRIBUTES THAT CORRESPONDS TO THE HYPERPARAMETER OF THE   ##
## MODEL. ALL OF THEM EXCEPT FOR THE LOSS FUNCTION AND THE ACTIVATION FUNCTION CAN BE OPTIMIZED THROUGH ## 
## THE Hype_Sel() METHOD.                                                                               ##
## TO PERFORM THE OPTIMIZATION OVER AN HYPERPARAMETER WE SET IT TO A LIST CONTAINING THE INTERVAL       ##
## OVER WHICH WE WANT TO RUN THE OPTIMIZATION, OR A SET OF DESCRETE VALUES OR STRINGS.                  ##
## THE METHOD build_opt_model() TAKES CARE OF BUILD THE OPTIMAL NETWORK THROUGH THE CLASS Net_flex AND  ##
## TO SET THE ATTRIBUTES OF THE HYPE_SELECT OBJECT TO THE OPTIMAL VALUES                                ##
##########################################################################################################

#IMPORT THE REQUIRED LIBRARIES
from numpy.core.numeric import NaN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use('agg')
import numpy as np
from Model_OPTUNA import Net_flex
import optuna
from optuna.trial import TrialState
import os 
from itertools import combinations as cmb
import inspect

#INSTANTIATE THE CLASS
class Hype_Select(nn.Module):
    def __init__(self, 
                train_data, 
                test_data, 
                n_layers_range = 2,
                n_units_range = 128,
                drop_out_range = 0.,
                n_epochs_range = 1000,
                batch_size_range = 1, 
                learning_rate_range = 0.001,
                optimizer = 'Adam', 
                loss_function: str = 'MSELoss',
                activation_func: str = 'ReLU',
                weight_decay_range = 0.,
                momentum_values = 0.):

        super(Hype_Select, self).__init__()

        ####################################### INPUT HANDLING #################################
        if not (isinstance(n_layers_range, int) or isinstance(n_layers_range,list)):
            raise TypeError('"n_layers_range" must be an integer or a list made by two integers')
        if isinstance(n_layers_range,list) and (len(n_layers_range) != 2):
            raise IndexError('"n_layers_range" must be of lenght 2')

        if not (isinstance(n_units_range, int) or isinstance(n_units_range,list)):
            raise TypeError('"n_units_range" must be an integer or a list made by two integers')
        if isinstance(n_units_range,list) and (len(n_units_range) != 2):
            raise IndexError('"n_units_range" must be of lenght 2')

        if not (isinstance(drop_out_range, float) or isinstance(drop_out_range,list)):
            raise TypeError('"drop_out_range" must be a float or a list made by two float')
        if isinstance(drop_out_range,list) and (len(drop_out_range) != 2):
            raise IndexError('"drop_out_range" must be of lenght 2')

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
        if not (isinstance(momentum_values, float) or isinstance(momentum_values,list)):
            raise TypeError('"momentum_values" must be a float value or a list made by two float values')
        if isinstance(momentum_values,list) and (len(momentum_values) != 2):
            raise IndexError('"momentum_values" must be of lenght 2')
        
        if not (isinstance(optimizer, str) or isinstance(optimizer,list)):
            raise TypeError('"optimizer" must be of type string or list')

        if not isinstance(loss_function,str):
            raise TypeError('"loss_function" must be of type string')
        
        if not isinstance(activation_func,str):
            raise TypeError('"activation_func" must be of type string')

        ################ OBJECT INITIALIZATION ################
        self.train_data = train_data
        self.test_data = test_data
        self.n_layers_range = n_layers_range
        self.n_units_range = n_units_range
        self.drop_out_range = drop_out_range
        self.n_epochs_range = n_epochs_range
        self.batch_size_range = batch_size_range
        self.learning_rate_range = learning_rate_range
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.weight_decay_range = weight_decay_range
        self.momentum_values = momentum_values
        self.activation_func = activation_func
        self.optimized_param = None
        self.model_arch = None

    
    def Hype_sel(self, study_name:str, n_trials, timeout, sampler=optuna.samplers.TPESampler(seed=10), direction='minimize'):

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
            train_loader = DataLoader(dataset=self.train_data, batch_size=batch_size, shuffle=True, num_workers=0)
            test_loader =  DataLoader(dataset=self.test_data, batch_size=len(self.test_data), shuffle=False, num_workers=0)

            #GENERATE THE MODEL
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            net = Net_flex(n_layers_range=self.n_layers_range, n_units_range=self.n_units_range, 
                            drop_out_range=self.drop_out_range, activation_func=self.activation_func)
            model = net.model(trial).to(device)
            
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
            
            #OPTIMIZER
            if isinstance(self.optimizer,list):
                opt = trial.suggest_categorical('optimizer',self.optimizer)
            else:
                opt = self.optimizer
                
            optimizer = getattr(optim, opt)(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            #MOMENTUM IF NEEDED
            if 'momentum' in inspect.getfullargspec(getattr(optim, opt))[0]:
                if isinstance(self.momentum_values, float):
                    momentum = self.momentum_values
                else:
                    momentum = trial.suggest_categorical('momentum', self.momentum_values)
                optimizer = getattr(optim, opt)(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

            #LOSS FUNCTION
            criterion = getattr(nn,self.loss_function)()

            #TRAINING LOOP
            for epoch in range(num_epochs):
                model.train()
                for i, (x, y) in enumerate(train_loader):
                    x = x.to(device)
                    y = y.to(device)
                    
                    #FORWARD PASS
                    pred = model(x)
                    loss = criterion(pred, y)

                    #BACKWARD PASS
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 

                #VALIDATION PROCESS
                model.eval() 
                test_loss = []
                with torch.no_grad():
                    for i, (x, y) in enumerate(test_loader):
                        x,y =  x.to(device), y.to(device)

                        pred = model(x)
                        loss = criterion(pred, y)
                        loss = loss.detach().cpu().numpy()
                        test_loss.append(loss)
                    
                    test_loss = np.mean(test_loss)
                    if test_loss == NaN:
                        test_loss = np.inf

                trial.report(test_loss, epoch)

                #HANDLE PRUNING PROCESS IN BASE OF THE TEST_LOSS VALUES.
                #THE PRUNER IS THE MEDIAN PRUNER, DEFAULT FOR OPTUNA.
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return  test_loss

        #CREATE AND RUN THE STUDY
        study = optuna.create_study(study_name=study_name, direction=direction, sampler=sampler, load_if_exists=True)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)


        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        
        #SELECT THE BEST TRIAL
        trial = study.best_trial

        #SET THE OPTIMIZER TO THE ONE OF THE BEST TRIAL
        if isinstance(self.optimizer,list):
            self.optimizer = trial.params['optimizer']

        #OUTPUT AND VISUALIZATION

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
        
        return 

    def build_opt_model(self):

        ################################################################################################
        ## THIS FUNCTION CAN RUN ONLY AFTER Hype_sel().This function can run only after Hype_sel().   ##
        ## IT SETS THE ATTRIBUTES OF AN Hype_select() OBJECT TO ITS OPTIMAL VALUE.                    ##
        ## IT INITIALIZES THE ATTRIBUTE 'model_arch' TO THE BEST ARCHITECTURE FOUND DURING THE STUDY, ##
        ## OR TO THE ONE SPECIFIED DURING THE INITIALIZATION                                          ##
        ################################################################################################
        
        if self.optimized_param is None:
            raise AttributeError("Before i can build the optimal model, you have to search for it!")
        
        if isinstance(self.batch_size_range,list):
            self.batch_size_range = self.optimized_param.get('batch size')

        if isinstance(self.n_epochs_range, list):
            self.n_epochs_range = self.optimized_param.get('number of epochs')

        if isinstance(self.learning_rate_range, list):
            self.learning_rate_range = self.optimized_param.get('learning rate')

        if isinstance(self.weight_decay_range, list):
            self.weight_decay_range = self.optimized_param.get('l2 regularization')

        if isinstance(self.momentum_values,list):
            self.momentum_values = self.optimized_param.get('momentum')

        if isinstance(self.n_layers_range, list):
            self.n_layers_range = self.optimized_param.get('n_layers')

        if isinstance(self.n_units_range, list): 
            units = []
            for i in range(self.n_layers_range):
                units.append(self.optimized_param.get('n_units_l{}'.format(i)))
        else:
            units = [self.n_units_range]*self.n_layers_range

        if isinstance(self.drop_out_range, list): 
            n_drop = []
            for i in range(self.n_layers_range):
                n_drop.append(self.optimized_param.get('dropout_l{}'.format(i)))
        else:
            n_drop = [self.drop_out_range]*self.n_layers_range
        
        input_features = getattr(Net_flex(1,1,1,self.loss_function),'input_features')
        output_features = getattr(Net_flex(1,1,1,self.loss_function),'output_features')
        #print(units)
        layers = []
        for i in range(self.n_layers_range):
            layers.append(nn.Linear(input_features, units[i]))
            layers.append(getattr(nn, self.activation_func)())
            if n_drop[i]!= 0.:
                layers.append(nn.Dropout(n_drop[i]))
            
            input_features = units[i]

        layers.append(nn.Linear(input_features, output_features))

        self.model_arch = nn.Sequential(*layers)

        return 




        

        