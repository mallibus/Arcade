#
import os
import numpy as np
import random as rn
import environment
import brain
import dqn
from matplotlib import pyplot as plt

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING PARAMETERS
epsilon  = 0.3
number_actions = 5
direction_boundary = (number_actions - 1)/2
number_epochs = 1000
max_memory = 3000
batch_size = 512
temperature_step = 1.5

# BUILD THE ENVIRONMENT

env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# BUILD THE BRAIN

brain = brain.Brain(learning_rate=0.00001, number_actions=number_actions)

# DQN MODEL

dqn = dqn.DQN(max_memory=max_memory,discount_factor=0.9)

# CHOOSING MODE
train = True 

# TRAINING
env.train = train
model = brain.model
if(env.train):
    reward_list = []
    for epoch in range(1,number_epochs):
        total_reward = 0
        loss = 0.
        new_month = np.random.randint(0,12) # iniziamo un mese a caso
        env.reset(new_month=new_month)
        game_over = False
        current_state,_,_ = env.observe()
        timestamp = 0
        # timestamp = 1 minuto
        # mi fermo dopo 5 mesi
        while ((not game_over) and timestamp <= 5*30*24*60):
            # PLAY NEXT ACTION BY EXPLORATION
            if np.random.rand() <= epsilon:
                action = np.random.randint(0,number_actions)   # indice dell'azione che facciamo
            # PLAY NEXT ACTION BY INFERENCE
            else:
                q_values = model.predict(current_state)
                action = np.argmax(q_values[0]) 
                
            if (action - direction_boundary < 0):
                direction = -1
            else:
                direction = +1
            energy_ai = abs(action - direction_boundary) * temperature_step    
            
            # UPDATE ENVIRONMENT
            next_state, reward, game_over = env.update_env(direction,energy_ai,int(timestamp/(30*24*60)))
            total_reward += reward
                
            # STORE THIS TRANSACTION INTO MEMORY
            dqn.remember([current_state,action,reward,next_state],game_over)
            
            # GET BATCHES INPUTS AND TARGETS
            inputs, targets = dqn.get_batch(model,batch_size = batch_size)
            
            # COMPUTING THE LOSS
            loss += model.train_on_batch(inputs,targets)
            timestamp += 1
            current_state = next_state
        
        #PRINT THE TRAINING RESULTS
        print("\n")
        print("Epoch: {:03d}/{:03d}".format(epoch,number_epochs))
        print("Total energy spent with AI: {:.0f}".format(env.total_energy_ai))
        print("Total energy spent without: {:.0f}".format(env.total_energy_noai))
        print("Total reward: {:.2f}".format(total_reward))
        reward_list.append(total_reward)
        if epoch % 10 ==0:
            plt.plot(reward_list)
            plt.show()
        
        #SAVE THE MODEL
        import datetime as dt
        filename = "model_"+dt.datetime.now().strftime("%Y%m%d_%H%M%S")+".h5"
        model.save(filename)