#######################################################################################
## THIS CLASS IMPLEMENTS THE MEMORY OF OUR AGENT.                                    ##
## IT STORES A GIVEN NUMBER OF STATES OF THE ENVIRONMENT AND RESPONSES OF THE AGENT. ##
## IT IS EXPLOITED TOGETHER WITH THE TARGET NETWORK TO PERFORM THE OPTIMIZATION.     ##
#######################################################################################

#IMPORT REQUIRED LIBRARIES
from collections import deque
import random

#INSTANTIATE THE CLASS
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity) # Define a queue with maxlen "capacity"

    def push(self, state, action, next_state, reward):
        self.memory.append( (state, action, next_state, reward) ) # Add the tuple (state, action, next_state, reward) to the queue

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self)) # Get all the samples if the requested batch_size is higher than the number of sample currently in the memory
        return random.sample(self.memory, batch_size) # Randomly select "batch_size" samples

    def __len__(self):
        return len(self.memory) # Return the number of samples currently stored in the memory