#
# Deep Q-Learning process
#
import numpy as np

# DEEP Q-LEARNING WITH EXPERIENCE REPLAY

class DQN(object):
    # PARAMS DEFINITION
    def __init__(self, max_memory = 100, discount_factor = 0.9):
        self.memory = list()
        self.max_memory = max_memory
        self.discount = discount_factor
        
        
    # METHOD TO STORE TRANSITIONS IN MEMORY
    def remember(self,transition, game_over):
        self.memory.append([transition,game_over])
        if len(self.memory)>self.max_memory:
            del self.memory[0]

    # MAKING A METHOD THAT BUILDS TWO BATCHES OF INPUT & TARGETS
    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        # numbe rof inputs = number of elements in the state
        num_inputs = self.memory[0][0][0].shape[1]   # memory[0]          - take first element of the memory 
                                                     # memory[0][0]       - take transition (first element in [transition, game_over])
                                                     # memory[0][0][0]    - transition is [current_state, action_played, reward, next_state]
                                                     # memory[0][0][0].shape[1] - number of elements in the matrix of states
                                                     # defined in environment.py  as next_state = np.matrix([scaled_temperature_ai,scaled_number_users,scaled_rate_data])
        num_outputs = model.output_shape[-1]
        
        inputs  = np.zeros((min(batch_size,len_memory), num_inputs))
        targets = np.zeros((min(batch_size,len_memory), num_outputs))
        
        for i,idx in enumerate(np.random.randint(0,len_memory, size = min(batch_size,len_memory))):
            current_state,action,reward,next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            inputs[i] = current_state
            targets[i] = model.predict(current_state)[0]
            Q_sa = np.max(model.predict(next_state)[0])
            if game_over:
                targets[i,action] = reward
            else:
                targets[i,action] = reward + self.discount * Q_sa
        return inputs, targets
    
    
            
        