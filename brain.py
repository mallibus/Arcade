# Import libs
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# BUILDING THE BRAIN

class Brain(object):
    
    def __init__(self, status_size = 6, learning_rate = 0.001, number_actions = 9):
        self.learning_rate = learning_rate
        states = Input(shape=(status_size,)) # Number of neurons to encode the state (status_size rows, 1 column)
        x = Dense(units=256, activation='sigmoid')(states)
        x = Dropout(rate = 0.1)(x)

        y = Dense(units=64, activation='sigmoid')(x)
        y = Dropout(rate = 0.1)(x)
        q_values = Dense(units = number_actions, activation='softmax')(y)
        
        self.model = Model(inputs = states, outputs = q_values)
        self.model.compile(loss = 'mse',optimizer= Adam(lr=learning_rate))

        self.loss_trend = []

        print("Created Brain")
        print(self.model.summary())
        
        
        