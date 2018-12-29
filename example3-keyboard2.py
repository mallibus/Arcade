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

# --- Constants ---
SPRITE_SCALING_PLAYER = 0.5
SPRITE_SCALING_COIN = 0.2
SPRITE_SCALING_BALL = 0.5
EXPLOSION_TEXTURE_COUNT = 60

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

BALLS_COUNT = 1
COINS_COUNT = 2
COINS_MAX_SPEED = 4
MOVEMENT_SPEED = 5


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

        self.center_x = 50
        self.center_y = 50

        self.change_x = 0
        self.change_y = 0

    def update(self):
        self.center_x += self.change_x
        self.center_y += self.change_y

        if self.left < 0:
            self.left = 0
        elif self.right > SCREEN_WIDTH - 1:
            self.right = SCREEN_WIDTH - 1

        if self.bottom < 0:
            self.bottom = 0
        elif self.top > SCREEN_HEIGHT - 1:
            self.top = SCREEN_HEIGHT - 1

class MyGame(arcade.Window):
    """ Our custom Window Class"""

    def __init__(self):
        """ Initializer """
        # Call the parent class initializer
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, "Sprite Example")

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

        # Track the current state of what key is pressed
        self.left_pressed = False
        self.right_pressed = False
        self.up_pressed = False
        self.down_pressed = False

        arcade.set_background_color(arcade.color.AMAZON)

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


    def on_update(self, delta_time):
        """ Movement and game logic """
        if not self.game_over:
            # Calculate speed 
            self.player.change_x , self.player.change_y = self.pick_keyboard_action()

            # Call update on all sprites (The sprites don't do much in this
            # example though.)
            self.all_sprites_list.update()

            reward, self.game_over = self.update_env_get_reward()
            self.score += reward
        else:
            self.setup()







def main():
    window = MyGame()
    window.setup()
    arcade.run()
    print("Game Over")


if __name__ == "__main__":
    main()