############################################################################################
## IN THIS FILE WE FIND A SCRIPT THAT PERFORM THE TRAINING WITH THE HYPERPARAMETERS FOUND ##
## DURING THE OPTIMIZATION PROCEDURE AND PRODUCES SOME USEFUL PLOTS.                      ##
############################################################################################

#IMPORT REQUIRED LIBRARIES
import torch
import gym
import numpy as np
from policies import choose_action_softmax 
from replay_memory import ReplayMemory
from policy_net import DQN, update_step
import random
from torch import nn
import matplotlib.pyplot as plt

#INITIALIZE THE GYM ENVIRONMENT
env = gym.make('CartPole-v1') 
env = gym.wrappers.Monitor(env, './video',video_callable= lambda i: i%100==0, force=True)

#REPRODUCIBILITY
env.seed(0) 
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

#HYPERPARAMETERS
gamma = 0.9685315043284639  # gamma parameter for the long term reward
replay_memory_capacity = 10000   # Replay memory capacity
lr = 0.07985216521144392   # Optimizer learning rate
target_net_update_steps = 3   # Number of episodes to wait before updating the target network
batch_size = 128   # Number of samples to take from the replay memory for each update
bad_state_penalty = -1  # Penalty to the reward when we are in a bad state (in this case when the pole falls down) 
min_samples_for_training = 1000   # Minimum samples in the replay memory to enable the training

#INSTANTIATE THE AGENT
replay_mem = ReplayMemory(replay_memory_capacity)
policy_net = DQN(state_space_dim=4,action_space_dim=2)
target_net = DQN(state_space_dim=4,action_space_dim=2)
target_net.load_state_dict(policy_net.state_dict())

#EXPONENTIALLY DECAYING EXPLORATION PROFILE
initial_value = 8
num_iterations = 1000
exp_decay = np.exp(-np.log(initial_value) / num_iterations * initial_value ) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

#OPTIMIZER
optimizer = torch.optim.SGD(policy_net.parameters(), lr=lr) 

#LOSS FUNCTION
loss_fn = nn.SmoothL1Loss()

#LOG FOR THE SCORES
score_history = []

#TRAINING LOOP
for episode_num, tau in enumerate(exploration_profile):
    # Reset the environment and get the initial state
    state = env.reset()
    # Reset the score. The final score will be the total amount of steps before the pole falls
    score = 0
    done = False
    # Go on until the pole falls off
    while not done:
        # Choose the action following the policy
        action, q_values = choose_action_softmax(policy_net, state, temperature=tau)
      
        # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
        next_state, reward, done, info = env.step(action)

        # We apply a (linear) penalty when the cart is far from center
        pos_weight = 1
        ang_vel_weight = 1
        reward = reward - pos_weight * np.abs(state[0]) - ang_vel_weight*np.abs(state[3])**2

        # Update the final score (+1 for each step)
        score += 1

        # Apply penalty for bad state
        if done: # if the pole has fallen down 
            reward += bad_state_penalty
            next_state = None

        # Update the replay memory
        replay_mem.push(state, action, next_state, reward)

        # Update the network
        if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
            update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size)

        # Set the current state for the next iteration
        state = next_state

    # Update the target network every target_net_update_steps episodes
    if episode_num % target_net_update_steps == 0:
        print('Updating target network...')
        target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
    
    score_history.append(score)
    # Print the final score
    print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}")

#COMPUTE THE AVERAGE SCORE
score_mean = np.mean(np.array(score_history))

#COMPUTE THE EPISODE AFTER WHICH THE AVERAGE SCORE IS GREATER THAN 490
convergence_episode = num_iterations
if 500 in score_history:
    convergence = False
    ii = score_history.index(500)
    while not convergence and ii < num_iterations:
        if np.mean(np.array(score_history)[ii:]) > 490:
            convergence_episode = ii
            convergence = True
        else:
          ii += 1
env.close()

#PLOT THE SCORES HISTORY
plt.figure(figsize=(12,10))
plt.plot(score_history,label='Episode score')
plt.axvline(x=convergence_episode, color='r',ls='--', label=f'Convergence episode = {convergence_episode}')
plt.plot([score_mean for i in range(1000)],label=f'Average score = {score_mean}')
plt.xlabel('episode')
plt.ylabel('score')
plt.legend()
plt.savefig('score_history.pdf')

#PLOT THE EXPLORATION PROFILE
plt.figure(figsize=(12,10))
plt.plot(exploration_profile,label='Temperature')
plt.xlabel('episode')
plt.ylabel('temperature')
plt.legend()
plt.savefig('exploration_profile.pdf')