from __future__ import division

import random

from backgammon.game import Game
from backgammon.agents.human_agent import HumanAgent
from backgammon.agents.random_agent import RandomAgent
from backgammon.agents.ai_agent import TDAgent

from subnet import *


class MonoNN:
    def __init__(self, model_path, summary_path, checkpoint_path, restore=False):
        g3 = tf.Graph()
        s3 = tf.Session(graph=g3)
        with s3.as_default(), g3.as_default():
            self.mono_nn = MonolithicNet(s3, model_path, summary_path, checkpoint_path, restore)

    # this method is not really related to the model but it is encapsulated as part of the model class
    def play(self):
        game = Game.new()
        game.play([HumanAgent(Game.TOKENS[0]), TDAgent(Game.TOKENS[1], self)], draw=True)

    def test(self, episodes=100, draw=False):
        players = [TDAgent(Game.TOKENS[0], self), RandomAgent(Game.TOKENS[1])]
        winners = [0, 0]
        for episode in range(episodes):
            game = Game.new()

            winner = game.play(players, draw=draw)
            winners[winner] += 1

            winners_total = sum(winners)
            print("[Episode %d] %s (%s) vs %s (%s) %d:%d of %d games (%.2f%%)" % (episode,
                players[0].player, players[0].player,
                players[1].player, players[1].player,
                winners[0], winners[1], winners_total,
                (winners[0] / winners_total) * 100.0))

    def get_output(self, x):
        return self.mono_nn.get_output(x)

    def train(self):
        self.mono_nn.create_model()

        # the agent plays against itself, making the best move for each player
        players = [TDAgent(Game.TOKENS[0], self), TDAgent(Game.TOKENS[1], self)]

        validation_interval = 1000
        episodes = 5000

        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0:
                self.test(episodes=100)

            game = Game.new()
            player_num = random.randint(0, 1)

            x = game.extract_features(players[player_num].player)

            game_step = 0
            while not game.is_over():
                roll = game.roll_dice()
                if player_num:
                    game.reverse()
                game.take_turn(players[player_num], roll, nodups=True)
                if player_num:
                    game.reverse()
                game.next_step(players[player_num], player_num)
                player_num = (player_num + 1) % 2

                x_next = game.extract_features(players[player_num].player)
                V_next = self.get_output(x_next)

                self.mono_nn.run_output(x, V_next)

                x = x_next
                game_step += 1

            winner = game.winner()

            self.mono_nn.update_model(x, winner,episode, episodes, players, game_step)

        self.mono_nn.training_end()

        self.test(episodes=1000)
