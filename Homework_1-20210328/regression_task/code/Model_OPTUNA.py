#######################################################################################
## THIS CLASS DEFINE THE NN-MODEL IN A DYNAMIC WAY, SUCH THAT IT IS ABLE TO INTERACT ##
## AND TO RECEIVE INPUTS FROM THE HYPE_SELECT FUNCTION                               ##
#######################################################################################

#IMPORT THE REQUIRED LIBRARIES
import torch.nn as nn

#INSTANTIATE THE CLASS
class Net_flex(nn.Module):
    def __init__(self,
                 n_layers_range : list or int,
                 n_units_range : list or int ,
                 drop_out_range : list or float,
                 activation_func : str):
        super(Net_flex, self).__init__()
        self.n_layers = 1
        self.n_layers_range = n_layers_range
        self.n_units_range = n_units_range
        self.drop_out_range = drop_out_range
        self.layers = []
        self.input_features = 1
        self.output_features = 1
        self.activation_func = activation_func

    
    #METHOD THAT BULD THE NETWORK WITH THE PARAMETERS SELECTED BY THE USER 
    #OR WITH THE PARAMETER SAMPLED BY THE OPTUNA SAMPLER DURING THE OPTIMIZATION PROCEDURE
    def model(self, trial):
        input_features = self.input_features
        if isinstance(self.n_layers_range, int):
            self.n_layers = self.n_layers_range
        else:
            self.n_layers = trial.suggest_int("n_layers", self.n_layers_range[0],self.n_layers_range[1])
        
        for i in range(self.n_layers):
            if isinstance(self.n_units_range, tuple) and isinstance(self.n_layers_range, int) and len(self.n_units_range)==self.n_layers_range:
                out_features = self.n_units_range[i]
            else:
                out_features = trial.suggest_int("n_units_l{}".format(i), self.n_units_range[0], self.n_units_range[1])

            self.layers.append(nn.Linear(input_features, out_features))
            self.layers.append(getattr(nn, self.activation_func)())
            if isinstance(self.drop_out_range, float):
                p = self.drop_out_range
            else:
                 p = trial.suggest_float("dropout_l{}".format(i), self.drop_out_range[0], self.drop_out_range[1])
            self.layers.append(nn.Dropout(p))
            
            input_features = out_features
    
        self.layers.append(nn.Linear(input_features, self.output_features))

        return nn.Sequential(*self.layers)


