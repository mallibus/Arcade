# Import libs
import os
import datetime as dt
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# BUILDING THE BRAIN

class Brain(object):
    
    def __init__(self, status_size = 6, learning_rate = 0.001, number_actions = 9, model_file=""):
        self.learning_rate = learning_rate
        states = Input(shape=(status_size,)) # Number of neurons to encode the state (status_size rows, 1 column)

        x = Dense(units=128, activation='sigmoid')(states)
        x = Dropout(rate = 0.2)(x)

        y = Dense(units=32, activation='sigmoid')(x)
        y = Dropout(rate = 0.2)(y)
        q_values = Dense(units = number_actions, activation='softmax')(y)
        
        self.model = Model(inputs = states, outputs = q_values)
        self.model.compile(loss = 'mse',optimizer= Adam(lr=learning_rate))

        # keep track of loss during training
        self.loss_trend = []

        if len(model_file)>0:
            if os.path.isfile(model_file):
                print("{:s} model file loaded".format(model_file))
                self.model = load_model(model_file)
            else:
                print("{:s} model file not found - starting from scratch".format(model_file))

        print("Brain ready:")
        print(self.model.summary())
    
    def learn_batch(self,inputs,targets,loss_file = "", model_file =""):
        loss = self.model.train_on_batch(inputs,targets)
        self.loss_trend.append(loss)
        if len(loss_file)>0:
            self.save_loss(loss_file)
        if len(model_file)>0:
            self.model.save(model_file)
        # Save inputs and targets for the batch for future analyisis
        np.savetxt("00in.txt",inputs,delimiter=',')
        np.savetxt("00target.txt",targets,delimiter=',')
        return loss

    def save_loss(self,loss_file):
        exists = os.path.isfile(loss_file)
        if not exists:
            print("Creating loss file with header")
            header = "RL Brain - Loss file -  " + dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "\n"
            header = header + "Loss\n"
        else:
            header=""
        with open(loss_file,"a+") as f:
            f.write(header)
            if len(self.loss_trend)>0:
                f.write(str(self.loss_trend[-1])+"\n")
    
    # Compute how much loss is decreased from the max loss experienceds to the average of the last samples
    # This is used as "epsilon" in DQL
    def loss_decay(self,samples=50):
        if len(self.loss_trend)<(samples):
            print("decay too early :", len(self.loss_trend))
            return 1.0
        big_loss = np.max(self.loss_trend)
        last_loss = np.mean(self.loss_trend[-samples:])
        decay = (1-(big_loss-last_loss)/big_loss)
        print("decay : ",decay)
        return decay
        