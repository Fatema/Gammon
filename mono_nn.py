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

    def get_output(self, x):
        return '', self.mono_nn.get_output(x)

    def restore_previous(self):
        self.mono_nn.restore_previous()

    def print_checkpoints(self):
        self.mono_nn.print_checkpoints()

    def extract_features(self, game, player):
        features = []
        # the order in which the players are evaluated matters
        for k in range(len(game.players)):
            p = game.players[k]
            pip_count = 0
            hit_count = 0
            for j in range(len(game.grid)):
                col = game.grid[j]
                feats = [0.] * 4
                if len(col) > 0 and col[0] == p:
                    if len(col) == 1: hit_count += 1
                    if k == game.reversed:
                        temp = len(col) * (j + 1)
                        pip_count += temp
                    else:
                        temp = len(col) * (24 - j)
                        pip_count += temp
                    # set the features to be 4 units each, last unit is set to (n-3)/2
                    for i in range(len(col)):
                        if i >= 3: break
                        feats[i] += 1
                    feats[3] = (len(col) - 3) / 2. if len(col) > 3 else 0
                features += feats
            # td gammon had it like this to scale the range between 0 and 1
            features.append(float(len(game.bar_pieces[p])) / 2.)
            features.append(float(len(game.off_pieces[p])) / game.num_pieces[p])
            # pip_count for the player the closer to home the less the value is
            pip_count += len(game.bar_pieces[p]) * 24
            features.append(float(pip_count))
        if player == game.players[0]:
            features += [1., 0.]
        else:
            features += [0., 1.]
        return np.array(features).reshape(1, -1)

    def play(self):
        game = Game.new()
        game.play([TDAgent(Game.PLAYERS[0], self), HumanAgent(Game.PLAYERS[1])], draw=True)

    def train(self, episodes=5000):
        self.mono_nn.create_model()

        # the agent plays against itself, making the best move for each player
        players = [TDAgent(Game.PLAYERS[0], self), TDAgent(Game.PLAYERS[1], self)]

        validation_interval = 100

        self.mono_nn.set_test_checkpoint()

        for episode in range(episodes):
            # print()
            # print()
            # print('episode', episode)
            if episode != 0 and episode % validation_interval == 0:
                tester.test_self(self)
                self.mono_nn.set_previous_checkpoint()
                self.mono_nn.set_test_checkpoint()
                tester.test_random(self)
                # self.print_checkpoints()

            game = Game.new()
            player_num = random.randint(0, 1)

            x = self.extract_features(game, players[player_num].player)

            # print('game beginning ...')
            # game.draw_screen()

            game_step = 0
            while not game.is_over():
                # game.draw_screen()

                game.next_step(players[player_num])
                player_num = (player_num + 1) % 2

                x_next = self.extract_features(game, players[player_num].player)
                V_next = self.mono_nn.get_output(x_next)

                self.mono_nn.run_output(x, V_next)

                x = x_next

            winner = game.winner()
            gammon_win = game.check_gammon(winner)
            out = np.array([[winner, gammon_win and winner, gammon_win and not winner]], dtype='float')

            print("[Train %d/%d] (winner: '%s') in %d turns" % (episode, episodes,
                                                                players[winner].player,
                                                                game.num_steps))

            self.mono_nn.update_model(x, out, episode, episodes, players, game_step)

        self.mono_nn.training_end()

        tester.test_self(self)
