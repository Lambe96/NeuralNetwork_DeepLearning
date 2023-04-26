#################################################################################
## IN THIS FILE WE FIND:                                                       ##
## - THE CLASS THAT DEFINES THE ARCHITECTURE OF OUR POLICY AND TARGET NETWORKS ##
## - THE FUNCTION THAT DEFINES THE OPTIMIZATION STEP                           ##
#################################################################################

#IMPORT THE REQUIRED LIBRARIES
import torch
from torch import nn
import numpy as np

#INSTANTIATE THE CLASS THAT DEFINES THE DEEP Q NETWORKS USED TO APPROXIMATE THE OPTIMAL POLICY 
class DQN(nn.Module):

    def __init__(self, state_space_dim, action_space_dim):
        super().__init__()

        self.linear = nn.Sequential(
                nn.Linear(state_space_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, action_space_dim)
                )

    def forward(self, x):
        return self.linear(x)


#DEFINE THE FUNCTION THAT DEFINE THE OPTIMIZATION STEP
def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size):
        
    # Sample the data from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)
    # Create tensors for each element of the batch
    states      = np.array([s[0] for s in batch])
    states      = torch.tensor(states).type(torch.float)
    actions     = np.array([s[1] for s in batch])
    actions     = torch.tensor(actions, dtype=torch.int64)
    rewards    = np.array([s[3] for s in batch])
    rewards     = torch.tensor(rewards, dtype=torch.float32)

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = np.array([s[2] for s in batch if s[2] is not None])
    non_final_next_states = torch.tensor(non_final_next_states, dtype=torch.float32) # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)

    # Compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states)
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1))

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states)
    next_state_max_q_values = torch.zeros(batch_size)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0]

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1) # Set the required tensor shape

    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()