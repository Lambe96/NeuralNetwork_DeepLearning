###########################################################################################################################
## IN THIS SCRIPT WE FIND AN OPTUNA STUDY OVER SOME HYPERPARAMETERS OF THE MODEL                                         ##
## THE STUDY IS A MULTY-OBJECTIVE ONE, THAT SEARCH THE CONFIGURATION THAT MAXIMIZES THE AVERAGE SCORE OVER 1000 EPISODES ##
## AND THAT MINIMIZES THE NUMBER OF EPISODES NEEDED TO REACH AND MAINTAIN A SCORE GREATER THAN 450                       ##
###########################################################################################################################

#IMPORT REQUIRED LIBRARIES
import numpy as np
from policy_net import DQN, update_step
import torch
from torch import nn
import gym
from policies import choose_action_softmax
from replay_memory import ReplayMemory
import optuna
import os
from itertools import combinations as cmb

#SET UP THE FIXED HYPERPARAMETERS
state_space_dim = 4
action_space_dim = 2
num_iterations = 1000
min_samples_for_training = 1000

#DEFINE THE OBJECTIVE FUNCTION
def objective(trial):
  
  #INITIAL TEMPERATURE OF THE SOFTMAX DISTRIVUTION
  initial_temp = trial.suggest_int("max_temp", 4, 8)

  #DISCOUNT FACTOR GAMMA
  gamma = trial.suggest_float("gamma", 0.96, 0.99)

  #NUMBER OF TIME STEPS BETWEEN TWO UPDATES OF THE TARGET NETWORK
  target_net_update_steps = trial.suggest_categorical("target_net_update_steps", [3,5,10,15])

  #LEARNING RATE
  learning_rate = trial.suggest_float('learning_rate',0.001,0.1)

  #PENALTY FOR BAD STATES
  bad_state_penalty = -1
  
  #FIXED BATCH SIZE
  batch_size = 128

  #INSTANTIATE THE AGENT
  replay_memory = ReplayMemory(10000) 

  policy_net = DQN(state_space_dim, action_space_dim)
  target_net = DQN(state_space_dim, action_space_dim)
  target_net.load_state_dict(policy_net.state_dict())
  optimizer = torch.optim.SGD(policy_net.parameters(), lr=learning_rate)
  criterion = nn.SmoothL1Loss()
  
  #INSTANTIATE THE ENVIRONMENT
  env = gym.make('CartPole-v1') 
  env.seed(42)
  
  #EXPONENTIALLY DECAYING EXPLORATION PROFILE
  decay = np.exp(-np.log(initial_temp) / num_iterations * initial_temp)
  exploration_profile = [initial_temp * (decay ** i) for i in range(num_iterations)]
  
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
      reward = reward - pos_weight * np.abs(state[0]) 

      # Update the final score (+1 for each step)
      score += 1

      # Apply penalty for bad state
      if done: # if the pole has fallen down 
          reward += bad_state_penalty
          next_state = None

      # Update the replay memory
      replay_memory.push(state, action, next_state, reward)

      # Update the network
      if len(replay_memory) > min_samples_for_training: # we enable the training only if we have enough samples in the replay memory, otherwise the training will use the same samples too often
          update_step(policy_net, target_net, replay_memory, gamma, optimizer, criterion, batch_size)

      # Set the current state for the next iteration
      state = next_state

    # Update the target network every target_net_update_steps episodes
    if episode_num % target_net_update_steps == 0:
        #print('Updating target network...')
        target_net.load_state_dict(policy_net.state_dict()) # This will copy the weights of the policy network to the target network

    # Print the final score
    if episode_num%100==0:
        print(f"EPISODE: {episode_num + 1} - FINAL SCORE: {score} - Temperature: {tau}")
    
    score_history.append(score)

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
  #RETURN BOTH THE CONVERGENCE EPISODE AND THE SCORE MEAN
  return convergence_episode , score_mean

#50 TRIAL STUDY
study = optuna.create_study(study_name = 'study1',directions=["minimize","maximize"])
study.optimize(objective, n_trials=50, timeout=43000)

#SAVE THE CONFIGURATIONS AND THE RESULTS OF THE STUDY IN A TXT FILE
trials = study.trials
f = open('study_params.txt','w+')
for i in range(len(trials)):
    f.write(f'trial {i+1}\t values: {trials[i].values}\t params: {trials[i]._params}\n')
f.close()
dir = os.path.join(os.getcwd(),'Visualization')
if not os.path.exists(dir):
    os.mkdir(dir)
os.chdir(dir)

#PARETO PLOT OF THE STUDY
fig = optuna.visualization.plot_pareto_front(study,target_names=["convergence_episode","score_mean"])
fig.write_image("pareto.pdf")