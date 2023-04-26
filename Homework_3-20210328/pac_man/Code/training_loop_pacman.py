#####################################################################################
## IN THIS FILE WE FIND A SCRIPT THAT PERFORM THE TRAINING WITH THE SOFTMAX POLICY ##
## AND PRODUCES SOME USEFUL PLOTS.                                                 ##
#####################################################################################

#IMPORT THE REQUIRED LIBRARIES
import torch
import gym
import numpy as np
from policies_PIL import update_step, choose_action_softmax
from replay_memory import ReplayMemory
from policy_net_cuda import DQN_PIL
import random
from torch import nn
import matplotlib.pyplot as plt
#INITIALIZE THE GYM ENVIRONMENT
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
env = gym.make('MsPacman-v0') 

#RECORD ONE EPISODE EVERY 100
env = gym.wrappers.Monitor(env, './video',video_callable= lambda i: i%100==0, force=True)

#REPRODUCIBILITY
env.seed(0) 
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

#HYPERPARAMETERS
gamma = 0.99                    #DISCOUNT FACTOR
replay_memory_capacity = 50000  #REPLAY MEMORY CAPACITY
lr = 0.00025                    #LEARNING RATE
target_net_update_steps = 5     #NUMBER OF EPISODES BETWEEN TWO UPDATES OF THE TARGET NETWORK
batch_size = 48                 #SIZE OF THE BATCH SAMPLED FROM THE REPLAY MEMORY
bad_state_penalty = -1.         #PENALTY TO ASSIGN WHEN THE PACMAN DIES
min_samples_for_training = 2000 #MINIMUM SAMPLES IN REPLAY MEMORY TO ENABLE THE TRAINING
num_iterations = 4000           #NUMBER OF EPISODES
initial_value = 8               #INITIAL TEMPERATURE

#SET UP THE AGENT
replay_mem = ReplayMemory(replay_memory_capacity)
policy_net = DQN_PIL()
target_net = DQN_PIL()
target_net.load_state_dict(policy_net.state_dict())

#OPTIMIZER AND LOSS FUNCTION
optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr) 
loss_fn = nn.SmoothL1Loss()

#LOGS FOR THE SCORES AND REWARDS
score_history = []
reward_history = []

#EXPLORATION PROFILE
exp_decay = np.exp(-np.log(initial_value) / num_iterations * initial_value ) # We compute the exponential decay in such a way the shape of the exploration profile does not depend on the number of iterations
exploration_profile = [initial_value * (exp_decay ** i) for i in range(num_iterations)]

#TRAINING LOOP
for episode_num, tau in enumerate(exploration_profile):
    # Reset the environment and get the initial state
    state = env.reset()
    # Reset the score. The final score will be the total amount of steps before the pole falls
    score = 0
    done = False
    # Go on until the pole falls off
    lives=3
    cum_reward = 0
    while not done:
        # Choose the action following the SOFTMAX policy
        action, q_values = choose_action_softmax(policy_net, state, temperature=tau, device=device)
      
        # Apply the action and get the next state, the reward and a flag "done" that is True if the game is ended
        next_state, reward, done, info = env.step(action)

        # Update the final score (+1 for each step)
        score += reward


        # Apply a penalty each time the pacman gets eaten
        if info['ale.lives']!= lives:
            reward = bad_state_penalty

        reward = np.clip(reward,-1,1)
        cum_reward+=reward
        # Update the replay memory
        replay_mem.push(state, action, next_state, reward)
        # Update the network
        if len(replay_mem) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
            update_step(policy_net, target_net, replay_mem, gamma, optimizer, loss_fn, batch_size, device=device)
        
        state = next_state
        lives = info['ale.lives']

        # Set the current state for the next iteration
        state = next_state

    # Update the target network every target_net_update_steps episodes
    if episode_num % target_net_update_steps == 0:
        print('Updating target network...')
        target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network
    
    score_history.append(score)
    reward_history.append(cum_reward)
    # Print the final score
    print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau:.3f} ") # Print the final score
    
    #every 100 episodes produces some plots:
    if (episode_num+1) % 100 == 0:
        #plot of the scores
        plt.figure(figsize=(12,10))
        plt.plot(score_history,label='episode score')
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.legend()
        plt.savefig(f'score_history_{episode_num+1}_reduced.pdf')

        #moving average plot
        score_history_mean = [sum(score_history[i:i+target_net_update_steps])/target_net_update_steps for i in range(episode_num-target_net_update_steps)]
        plt.figure(figsize=(12,10))
        plt.plot(score_history_mean,label='average score of five consecutive episode')
        plt.xlabel('episode')
        plt.ylabel('score')
        plt.legend()
        plt.savefig(f'score_history_mean_{episode_num+1}_reduced.pdf')
        
        #plot of the rewards
        plt.figure(figsize=(12,10))
        plt.plot(reward_history,label='episode reward')
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.legend()
        plt.savefig(f'reward_history.pdf')

score_mean = np.mean(np.array(score_history))
env.close()
