from __future__ import division

import random

import tester
from backgammon.agents.ai_agent import TDAgent
from backgammon.agents.random_agent import RandomAgent
from backgammon.game import Game

from backgammon.agents.human_agent import HumanAgent
from subnet import *


class MonoNN:
    def __init__(self, model_path, summary_path, checkpoint_path, restore=False):
        self.mono_nn = SubNet()
        self.mono_nn.set_network_name('mono')
        self.mono_nn.set_paths(model_path, summary_path, checkpoint_path)
        self.mono_nn.start_session(restore=restore)

    # this method is not really related to the model but it is encapsulated as part of the model class
    def play(self):
        game = Game.new()
        game.play([HumanAgent(Game.TOKENS[0]), TDAgent(Game.TOKENS[1], self)], draw=True)

    def get_output(self, x):
        return '', self.mono_nn.get_output(x)

    def restore_previous(self):
        self.mono_nn.restore_previous()

    def restore_test_checkpoint(self, timestamp, game_number):
        self.mono_nn.restore_test_checkpoint(timestamp, game_number)

    def print_checkpoints(self):
        self.mono_nn.print_checkpoints()

    def train(self, episodes=5000):
        self.mono_nn.create_model()

        # the agent plays against itself, making the best move for each player
        players = [TDAgent(Game.TOKENS[0], self), TDAgent(Game.TOKENS[1], self)]

        validation_interval = 1000

        for episode in range(episodes):
            # print()
            # print()
            # print('episode', episode)
            if episode % validation_interval == 0:
                tester.test_self(self)
                tester.test_random(self)
                # self.print_checkpoints()

            game = Game.new()
            player_num = random.randint(0, 1)

            x = game.extract_features(players[player_num].player)

            # print('game beginning ...')
            # game.draw_screen()

            game_step = 0
            while not game.is_over():
                # print('Player', players[player_num].player, 'turn')
                # print('extracted features:', x)
                roll = game.roll_dice()
                # print('dice roll', roll)
                # game.draw_screen()
                if player_num:
                    game.reverse()

                game.take_turn(players[player_num], roll, nodups=True)

                if player_num:
                    game.reverse()

                player_num = (player_num + 1) % 2

                x_next = game.extract_features(players[player_num].player)
                # print('next features extracted', x_next)
                V_next = self.mono_nn.get_output(x_next)
                # print('next output', V_next)

                self.mono_nn.run_output(x, V_next)

                x = x_next
                game_step += 1

            winner = game.winner()

            print("[Train %d/%d] (Winner: %s) in %d turns" % (episode, episodes, players[not winner].player, game_step))

            self.mono_nn.update_model(x, winner)

        self.mono_nn.training_end()

        tester.test_self(self)
