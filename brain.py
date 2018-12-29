# Import libs
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# BUILDING THE BRAIN

class Brain(object):
    
    def __init__(self, learning_rate = 0.001, number_actions = 5):
        self.learning_rate = learning_rate
        states = Input(shape=(3,)) # Number of neurons to encode the state (3 rows, 1 column)
        x = Dense(units=64, activation='sigmoid')(states)
        y = Dense(units=32, activation='sigmoid')(x)
        q_values = Dense(units = number_actions, activation='softmax')(y)
        
        self.model = Model(inputs = states, outputs = q_values)
        self.model.compile(loss = 'mse',optimizer= Adam(lr=learning_rate))
        
        
        