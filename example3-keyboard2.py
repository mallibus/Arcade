"""
Sprite Collect Coins

Simple program to show basic sprite usage.

Artwork from http://kenney.nl

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.sprite_collect_coins_move_bouncing
"""

import random
import arcade
import numpy as np
import os
from tensorflow.keras.models import load_model
import datetime as dt


import random as rn
import brain
import dqn


# GAME --- Constants ---
SPRITE_SCALING_PLAYER = 0.5
SPRITE_SCALING_COIN = 0.2
SPRITE_SCALING_BALL = 0.8
EXPLOSION_TEXTURE_COUNT = 60

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400

HISTORY_FILE = "game_history.csv"
LOSS_FILE = "learning_loss.csv"

BALLS_COUNT = 0
COINS_COUNT = 20
COINS_MAX_SPEED = 0
MOVEMENT_SPEED = 5



# DQL . SETTING PARAMETERS
TRAINING = True
LOAD_MODEL = True
LEARNING_INTERVAL = 100
MODEL_FILE = "model.h5"

learning_rate=0.001
epsilon  = 0.5
max_memory = 3000
batch_size = 512
# Actions - 0 - stay, 1-8 move (see encode_action)
number_actions = 9
# status is the posizion of the player + the positions of other stuff
status_size = 2 + BALLS_COUNT*2 + COINS_COUNT*2


class Explosion(arcade.Sprite):
    """ This class creates an explosion animation """

    # Static variable that holds all the explosion textures
    explosion_textures = []

    def __init__(self, texture_list):
        super().__init__("my-images/explosion/explosion0000.png")

        # Start at the first frame
        self.current_texture = 0
        self.textures = texture_list

    def update(self):

        # Update to the next frame of the animation. If we are at the end
        # of our frames, then delete this sprite.
        self.current_texture += 1
        if self.current_texture < len(self.textures):
            self.set_texture(self.current_texture)
        else:
            self.kill()

class Coin(arcade.Sprite):

    def __init__(self, filename, sprite_scaling):

        super().__init__(filename, sprite_scaling)

        self.change_x = 0
        self.change_y = 0

        # Position the coin
        self.center_x = random.randrange(SCREEN_WIDTH)
        self.center_y = random.randrange(SCREEN_HEIGHT)
        self.change_x = random.randrange(-COINS_MAX_SPEED, COINS_MAX_SPEED+1)
        self.change_y = random.randrange(-COINS_MAX_SPEED, COINS_MAX_SPEED+1)

    def update(self):

        # Move the coin
        self.center_x += self.change_x
        self.center_y += self.change_y

        # If we are out-of-bounds, then 'bounce'
        if self.left < 0:
            self.change_x *= -1

        if self.right > SCREEN_WIDTH:
            self.change_x *= -1

        if self.bottom < 0:
            self.change_y *= -1

        if self.top > SCREEN_HEIGHT:
            self.change_y *= -1

class Ball(arcade.Sprite):

    def __init__(self, filename, sprite_scaling):

        super().__init__(filename, sprite_scaling)

        self.change_x = 0
        self.change_y = 0

        # Position the ball
        self.center_x = random.randrange(SCREEN_WIDTH)
        self.center_y = random.randrange(SCREEN_HEIGHT)
        self.change_x = random.randrange(4, 6)*random.choice([-1,1])
        self.change_y = random.randrange(4, 6)*random.choice([-1,1])

    def update(self):

        # Move the coin
        self.center_x += self.change_x
        self.center_y += self.change_y

        # If we are out-of-bounds, then 'bounce'
        if self.left < 0:
            self.change_x *= -1

        if self.right > SCREEN_WIDTH:
            self.change_x *= -1

        if self.bottom < 0:
            self.change_y *= -1

        if self.top > SCREEN_HEIGHT:
            self.change_y *= -1

class Player(arcade.Sprite):

    def __init__(self, filename, sprite_scaling):

        super().__init__(filename, sprite_scaling)

        self.center_x = int(SCREEN_WIDTH / 2)
        self.center_y = int(SCREEN_HEIGHT / 2)
        #self.center_x = random.randrange(SCREEN_WIDTH)
        #self.center_y = random.randrange(SCREEN_HEIGHT)

        self.change_x = 0
        self.change_y = 0

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y
        """
       # Limit player inside window
        if self.left < 0:
            self.left = 0
        elif self.right > SCREEN_WIDTH - 1:
            self.right = SCREEN_WIDTH - 1

        if self.bottom < 0:
            self.bottom = 0
        elif self.top > SCREEN_HEIGHT - 1:
            self.top = SCREEN_HEIGHT - 1
        """
       # Wrap player to other side (toroidal field)
        if self.center_x < 0:
            self.center_x = SCREEN_WIDTH - 1
        elif self.center_x > SCREEN_WIDTH - 1:
            self.center_x = 0

        if self.center_y < 0:
            self.center_y = SCREEN_HEIGHT - 1
        elif self.center_y > SCREEN_HEIGHT - 1:
            self.center_y = 0

class MyGame(arcade.Window):
    """ Our custom Window Class"""

    def __init__(self):
        """ Initializer """
        # Call the parent class initializer
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Sprite Example")

        # Set the working directory (where we expect to find files) to the same
        # directory this .py file is in. You can leave this out of your own
        # code, but it is needed to easily run the examples using "python -m"
        # as mentioned at the top of this program.
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)

        # Variables that will hold sprite lists
        self.all_sprites_list = None
        self.coin_list = None
        self.ball_list = None
        self.explosions_list = None

        # Set up the player info
        self.player = None
        self.score = 0

        self.game_over = False
        self.initialized = False
        self.timer = 0

        # Track the current state of what key is pressed
        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        arcade.set_background_color(arcade.color.AMAZON)
        
        # BUILD THE BRAIN
        self.brain = brain.Brain(status_size = status_size, learning_rate=learning_rate, number_actions=number_actions)
        if LOAD_MODEL:
            if os.path.isfile(MODEL_FILE):
                print("{:s} model file loaded".format(MODEL_FILE))
                self.brain.model = load_model("model.h5")
            else:
                print("{:s} model file not found - starting from scratch".format(MODEL_FILE))
                
        # DQN MODEL
        self.dqn = dqn.DQN(max_memory=max_memory,discount_factor=0.9)


    def setup(self):
        """ Set up the game and initialize the variables. """

        if not self.initialized:
            # Pre-load the animation frames. We don't do this in the __init__ because it
            # takes too long and would cause the game to pause.
            self.explosion_texture_list = []

            for i in range(EXPLOSION_TEXTURE_COUNT):
                # Files from http://www.explosiongenerator.com are numbered sequentially.
                # This code loads all of the explosion0000.png to explosion0270.png files
                # that are part of this explosion.
                texture_name = f"my-images/explosion/explosion{i:04d}.png"
                self.explosion_texture_list.append(arcade.load_texture(texture_name))

            self.initialized = True

        # Restores game status
        self.game_over = False
        self.timer = 0

        # Sprite lists
        self.all_sprites_list = arcade.SpriteList()
        self.coin_list = arcade.SpriteList()
        self.ball_list = arcade.SpriteList()
        self.explosions_list = arcade.SpriteList()

        # Score
        self.score = 0

        # Set up the player
        # Character image from kenney.nl
        self.player = Player("my-images/character.png", SPRITE_SCALING_PLAYER)
        self.all_sprites_list.append(self.player)

        for i in range(BALLS_COUNT):
            # Create the killer balls instances
            ball = Ball("my-images/pool_cue_ball.png", SPRITE_SCALING_BALL)
            self.all_sprites_list.append(ball)
            self.ball_list.append(ball)

        # Create the coins
        for i in range(COINS_COUNT):
            # Create the coin instance
            # Coin image from kenney.nl
            coin = Coin("my-images/coin_01.png", SPRITE_SCALING_COIN)
            # Add the coin to the lists
            self.all_sprites_list.append(coin)
            self.coin_list.append(coin)

        # Create game status history for learning
        self.game_history = []

    def on_draw(self):
        """ Draw everything """
        arcade.start_render()
        self.all_sprites_list.draw()

        # Put the text on the screen.
        output = f"Score: {self.score}"
        arcade.draw_text(output, 10, 20, arcade.color.WHITE, 14)

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """

        if key == arcade.key.UP:
            self.up_pressed = True
        elif key == arcade.key.DOWN:
            self.down_pressed = True
        elif key == arcade.key.LEFT:
            self.left_pressed = True
        elif key == arcade.key.RIGHT:
            self.right_pressed = True

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key. """

        if key == arcade.key.UP:
            self.up_pressed = False
        elif key == arcade.key.DOWN:
            self.down_pressed = False
        elif key == arcade.key.LEFT:
            self.left_pressed = False
        elif key == arcade.key.RIGHT:
            self.right_pressed = False

    def pick_keyboard_action(self):
        # Calculate speed based on the keys pressed
        change_x = 0
        change_y = 0

        if self.up_pressed and not self.down_pressed:
            change_y = MOVEMENT_SPEED
        elif self.down_pressed and not self.up_pressed:
            change_y = -MOVEMENT_SPEED
        if self.left_pressed and not self.right_pressed:
            change_x = -MOVEMENT_SPEED
        elif self.right_pressed and not self.left_pressed:
            change_x = MOVEMENT_SPEED
        return change_x,change_y

    def pick_dql_action(self):
        status_0 = self.get_current_status(scaled=True)
        q_values = self.brain.model.predict(np.matrix(status_0))
        action = np.argmax(q_values[0])
        action_encoder = { 0 : (0,0),\
                           1 : (0,1),\
                           2 : (1,1),\
                           3 : (1,0),\
                           4 : (1,-1),\
                           5 : (0,-1),\
                           6 : (-1,-1),\
                           7 : (-1,0),\
                           8:  (-1,1) }
        enc_action = action_encoder[action]
        change_x = MOVEMENT_SPEED * enc_action[0]
        change_y = MOVEMENT_SPEED * enc_action[1]
        return change_x,change_y

    def pick_random_action(self):
        action = np.random.randint(number_actions)
        action_encoder = { 0 : (0,0),\
                           1 : (0,1),\
                           2 : (1,1),\
                           3 : (1,0),\
                           4 : (1,-1),\
                           5 : (0,-1),\
                           6 : (-1,-1),\
                           7 : (-1,0),\
                           8:  (-1,1) }
        enc_action = action_encoder[action]
        change_x = MOVEMENT_SPEED * enc_action[0]
        change_y = MOVEMENT_SPEED * enc_action[1]
        return change_x,change_y
        
    def update_env_get_reward(self):

        reward = 0

        # Generate a list of coins that collided with the player.
        coin_hit_list = arcade.check_for_collision_with_list(self.player,
                                                             self.coin_list)
        # Loop through each colliding sprite, remove it, and add to the score.
        for coin in coin_hit_list:
            coin.kill()
            reward += 1
        # Kill coins hit by balls
        for ball in self.ball_list:
            coin_hit_list = arcade.check_for_collision_with_list(ball,self.coin_list)
            # Loop through each colliding sprite, remove it, and add to the score.
            for coin in coin_hit_list:
                coin.kill()                
        # Generate a list of balls that collided with the player.
        ball_hit_list = arcade.check_for_collision_with_list(self.player,
                                                             self.ball_list)
        for ball in ball_hit_list:
            reward -= 10
            explosion = Explosion(self.explosion_texture_list)
            explosion.center_x = ball.center_x
            explosion.center_y = ball.center_y
            self.explosions_list.append(explosion)
            self.all_sprites_list.append(explosion)
            ball.change_x += np.sign(ball.change_x)
            ball.change_y += np.sign(ball.change_y)

        return reward,len(self.coin_list)<=0

    def get_current_status(self,scaled = False):
        """Get the status of the game as an array of
           player,coins and opponents positions"""
        coins_coords = np.zeros(COINS_COUNT*2)
        for i,coin in enumerate(self.coin_list):
            if scaled:
                coins_coords[i*2]=coin.center_x / SCREEN_WIDTH
                coins_coords[i*2+1]=coin.center_y / SCREEN_HEIGHT
            else:
                coins_coords[i*2]=coin.center_x 
                coins_coords[i*2+1]=coin.center_y 
        balls_coords = np.zeros(BALLS_COUNT*2)
        for i,ball in enumerate(self.ball_list):
            if scaled:
                balls_coords[i*2]=ball.center_x / SCREEN_WIDTH
                balls_coords[i*2+1]=ball.center_y / SCREEN_HEIGHT
            else:
                balls_coords[i*2]=ball.center_x
                balls_coords[i*2+1]=ball.center_y

        if scaled:
            status = np.array([self.player.center_x/SCREEN_WIDTH, self.player.center_y/SCREEN_HEIGHT, *coins_coords, *balls_coords])
        else:
            status = np.array([self.player.center_x, self.player.center_y, *coins_coords, *balls_coords])
        return(status)


    def save_history(self):
        exists = os.path.isfile(HISTORY_FILE)
        if not exists:
            print("With header")
            header = "Arcade RL experiment - History file - " + dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "\n"           
            header = header + "learning_rate : {:f}\nepsilon : {:f}\nmax_memory : {:d}\nbatch_size : {:d}\n".\
                            format(learning_rate,epsilon,max_memory,batch_size)

            header = header + "s0_player_x,s0_player_y,"
            for i in range(COINS_COUNT):
                header = header + "s0_coin_{:d}_x,s0_coin_{:d}_y,".format(i,i)
            for i in range(BALLS_COUNT):
                header = header + "s0_ball_{:d}_x,s0_ball_{:d}_y,".format(i,i)
            header = header + "action,"
            header = header + "s1_player_x,s1_player_y,"
            for i in range(COINS_COUNT):
                header = header + "s1_coin_{:d}_x,s1_coin_{:d}_y,".format(i,i)
            for i in range(BALLS_COUNT):
                header = header + "s1_ball_{:d}_x,s1_ball_{:d}_y,".format(i,i)
            header = header + "game_over,reward\n"
        else:
            header=""

        print("Saving {:d} history records to {:s}".format(len(self.game_history),HISTORY_FILE))
        with open(HISTORY_FILE,"a+") as f:
            f.write(header)
            for l in self.game_history:
                txt = ",".join([str(x) for x in l])+'\n'
                f.write(txt)
        # Reset game historyu after saving
        self.game_history = []

    def save_loss(self,loss):
        exists = os.path.isfile(LOSS_FILE)
        if not exists:
            print("Loss file with header")
            header = "Arcade RL experiment - Loss file -  " + dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S") + "\n"
            header = header + "Loss\n"
        else:
            header=""
        with open(LOSS_FILE,"a+") as f:
            f.write(header)
            f.write(str(loss)+"\n")

    def encode_action(self):
        """ 9 possible actions: 0 - no movement
                                1 - move up
                                2 - move up-right
                                3 - mode right
                                4 - move down-right
                                5 - move down
                                6 - move down-left
                                7 - move left
                                8 - move up-left """
        cx = np.sign(self.player.change_x)
        cy = np.sign(self.player.change_y)
        action_decoder = {  (0,0)  : 0,\
                            (0,1)  : 1,\
                            (1,1)  : 2,\
                            (1,0)  : 3,\
                            (1,-1) : 4,\
                            (0,-1) : 5,\
                            (-1,-1): 6,\
                            (-1,0) : 7,\
                            (-1,1) : 8 }
        return action_decoder[(cx,cy)]

    def learn_something(self):
            # GET BATCHES INPUTS AND TARGETS
            inputs, targets = self.dqn.get_batch(self.brain.model,batch_size = batch_size)
            # COMPUTING THE LOSS
            loss = self.brain.model.train_on_batch(inputs,targets)
            self.save_loss(loss)
            print("Learning loss :",loss)
            self.brain.model.save(MODEL_FILE)

    def on_update(self, delta_time):
        """ Movement and game logic """
        # If game over saves the history of the game and restarts
        if  self.game_over:
            self.save_history()
            self.learn_something()
            self.setup()
        # Else go ahead with the game
        else:
            self.timer += 1
            status_0 = self.get_current_status(scaled=True)

            # Calculate speed 
            self.player.change_x , self.player.change_y = self.pick_keyboard_action()
            if(self.player.change_x==0)and(self.player.change_y==0):
                if TRAINING and (np.random.rand() <= epsilon):
                    #print("random")
                    self.player.change_x , self.player.change_y = self.pick_random_action()
                else:
                    #print("dql")
                    self.player.change_x , self.player.change_y = self.pick_dql_action()
            #print("x+=",self.player.change_x," y+=",self.player.change_y)

            # Call update on all sprites (The sprites don't do much in this
            # example though.)
            self.all_sprites_list.update()

            reward, self.game_over = self.update_env_get_reward()
            self.score += reward

            status_1 = self.get_current_status(scaled=True)

            self.game_history.append(np.array([*status_0,self.encode_action(),*status_1,int(self.game_over),reward]))
            self.dqn.remember([np.matrix(status_0),self.encode_action(),reward,np.matrix(status_1)],int(self.game_over))
 
            if (self.timer % LEARNING_INTERVAL)== 0:
                self.learn_something()

def main():
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()