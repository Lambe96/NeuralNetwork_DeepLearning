import torch
import random
import numpy as np
from torch import nn
from PIL import Image

def preprocess_observation(obs):
    img = Image.fromarray(obs)
    img = img.resize((84,84)).convert('L')  # resize and convert to grayscale
    img = np.array(img).reshape(1,84,84)
    return img

def choose_action_epsilon_greedy(net, state, epsilon,device):
    net.to(device)
    if epsilon > 1 or epsilon < 0:
        raise Exception('The epsilon value must be between 0 and 1')
                
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(preprocess_observation(state)).type(torch.float).unsqueeze(0).to(device) # Convert the state to tensor
        net_out = torch.flatten(net(state)).cpu()

    # Get the best action (argmax of the network output)
    best_action = int(net_out.argmax())
    # Get the number of possible actions
    action_space_dim = net_out.shape[-1]

    # Select a non optimal action with probability epsilon, otherwise choose the best action
    if random.random() < epsilon:
        # List of non-optimal actions
        non_optimal_actions = [a for a in range(action_space_dim) if a != best_action]
        # Select randomly
        action = random.choice(non_optimal_actions)
    else:
        # Select best action
        action = best_action
        
    return action, net_out.numpy()

def choose_action_softmax(net, state, temperature,device):
    net.to(device)
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')
        
    # If the temperature is 0, just select the best action using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return choose_action_epsilon_greedy(net, state, 0,device)
    
    # Evaluate the network output from the current state
    with torch.no_grad():
        net.eval()
        state = torch.tensor(preprocess_observation(state)).type(torch.float).unsqueeze(0).to(device)
        net_out = torch.flatten(net(state)).cpu()

    # Apply softmax with temp
    temperature = max(temperature, 1e-8) # set a minimum to the temperature for numerical stability
    softmax_out = nn.functional.softmax(net_out / temperature, dim=0).numpy()
      
    # Sample the action using softmax output as mass pdf
    all_possible_actions = np.arange(0, softmax_out.shape[-1])
    action = np.random.choice(all_possible_actions, p=softmax_out) # this samples a random element from "all_possible_actions" with the probability distribution p (softmax_out in this case)
    
    return action, net_out.numpy()

def update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size,device):
    policy_net.to(device)
    target_net.to(device)
    # Sample the data from the replay memory
    batch = replay_mem.sample(batch_size)
    batch_size = len(batch)
    # Create tensors for each element of the batch
    states      = np.array([preprocess_observation(s[0]) for s in batch])
    states      = torch.tensor(states).type(torch.float)
    actions     = np.array([s[1] for s in batch])
    actions     = torch.tensor(actions, dtype=torch.int64)
    rewards     = np.array([s[3] for s in batch])
    rewards     = torch.tensor(rewards, dtype=torch.float32)

    # Compute a mask of non-final states (all the elements where the next state is not None)
    non_final_next_states = np.array([preprocess_observation(s[2]) for s in batch if s[2] is not None])
    non_final_next_states = torch.tensor(non_final_next_states, dtype=torch.float32) # the next state can be None if the game has ended
    non_final_mask = torch.tensor([s[2] is not None for s in batch], dtype=torch.bool)
    # Compute all the Q values (forward pass)
    policy_net.train()
    q_values = policy_net(states.to(device)).cpu()
    # Select the proper Q value for the corresponding action taken Q(s_t, a)
    state_action_values = q_values.gather(1, actions.unsqueeze(1)).to(device)

    # Compute the value function of the next states using the target network V(s_{t+1}) = max_a( Q_target(s_{t+1}, a)) )
    with torch.no_grad():
      target_net.eval()
      q_values_target = target_net(non_final_next_states.to(device)).cpu()
    next_state_max_q_values = torch.zeros(batch_size)
    next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0]

    # Compute the expected Q values
    expected_state_action_values = rewards + (next_state_max_q_values * gamma)
    expected_state_action_values = expected_state_action_values.unsqueeze(1).to(device) # Set the required tensor shape

    # Compute the Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values).cpu()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Apply gradient clipping (clip all the gradients greater than 2 for training stability)
    nn.utils.clip_grad_norm_(policy_net.parameters(), 2)
    optimizer.step()