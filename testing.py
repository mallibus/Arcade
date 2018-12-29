# 
# Testing the AI

import os
import numpy as np
import random as rn
from keras.models import load_model
import environment

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(12345)

# SETTING PARAMETERS
number_actions = 5
direction_boundary = (number_actions - 1)/2
temperature_step = 1.5

# BUILD THE ENVIRONMENT

env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)

# LOADING PRE TRAINED BRAIN

#model = load_model("model_official.h5")
model = load_model("model_my3_70.h5")
model = load_model("model_my2_90.h5")


# CHOOSING MODE
train = False

# RUNNING A 1 YEAR SIMULATION IN INFERENCE MODE
env.train = train
current_state,_,_ = env.observe()

#for timestamp in range(0, 12*30*24*60):
for timestamp in range(0, 1*30*24*60):
            # PLAY NEXT ACTION BY INFERENCE
    q_values = model.predict(current_state)
    action = np.argmax(q_values[0])     
    if (action - direction_boundary < 0):
        direction = -1
    else:
        direction = +1
    energy_ai = abs(action - direction_boundary) * temperature_step    
            
    # UPDATE ENVIRONMENT
    next_state, _, _ = env.update_env(direction,energy_ai,int(timestamp/(30*24*60)))
    current_state = next_state
    if(timestamp%(24*60))==0:
        print("Day {:03d} : AI {:.0f} - No AI {:.0f} - Saving : {:4.2f}%\r".format(int(timestamp/(24*60)),
              env.total_energy_ai,env.total_energy_noai,100*(env.total_energy_noai - env.total_energy_ai)/env.total_energy_noai,
              end=''))

#PRINT THE TRAINING RESULTS
print("\n")
print("Total energy spent with AI: {:.0f}".format(env.total_energy_ai))
print("Total energy spent without: {:.0f}".format(env.total_energy_noai))
