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
        self.mono_nn.set_timestamp(int(time.time()))
        self.mono_nn.start_session(restore=restore)

    # this method is not really related to the model but it is encapsulated as part of the model class
    def play(self):
        game = Game.new(Game.LAYOUT)
        game.play([HumanAgent(Game.TOKENS[0]), TDAgent(Game.TOKENS[1], self)], draw=True)

    def get_output(self, x):
        return '', self.mono_nn.get_output(x)

    def restore_previous(self):
        self.mono_nn.restore_previous()

    def restore_test_checkpoint(self, timestamp, game_number):
        self.mono_nn.restore_test_checkpoint(timestamp, game_number)

    def print_checkpoints(self):
        self.mono_nn.print_checkpoints()

    def extract_features(self, game, player):
        features = []
        # print(player)
        # the order in which the players are evaluated matters
        for k in range(len(game.players)):
            p = game.players[k]
            pip_count = 0
            for j in range(len(game.grid)):
                col = game.grid[j]
                feats = [0.] * 6
                if len(col) > 0 and col[0] == p:
                    if k == 0:
                        temp = len(col) * (24 - j)
                        pip_count += temp
                        # print(p,'count per col', j, temp, pip_count, len(col))
                    else:
                        temp = len(col) * (j + 1)
                        pip_count += temp
                        # print(p,'count per col', j, temp, pip_count, len(col))
                    for i in range(len(col)):
                        if i >= 5: break
                        feats[i] += 1
                    feats[5] = (len(col) - 5) / 2. if len(col) > 5 else 0 # normalize the remaining pips
                features += feats
            # print('pip_count before off pieces', pip_count)
            features.append(float(len(game.off_pieces[p])) / game.num_pieces[p])
            features.append(float(len(game.bar_pieces[p])) / 2.)
            # print(game.bar_pieces[p], game.off_pieces[p])
            # if pip on the bar penalize the pip_count
            pip_count += len(game.bar_pieces[p]) * 25
            # pip_count for the player the closer to home the less the pip_count
            # scale it out or include it as part of the reward
            features.append(float(pip_count) / 167)
            # print('pip count for', p, pip_count)
        if player == game.players[0]:
            features += [1., 0.]
        else:
            features += [0., 1.]

        features = np.array(features).reshape(1, -1)

        features = self.add_hit_prob(features, game)

        return features

    # determine the hitting probability based on fields that include single checkers that are within the opponents reach
    # hit_count / max(num_pieces - off_pieces - bar_pieces, 1)
    def add_hit_prob(self, features, game):
        # make the indexes evaulated based on global variables
        flip = features[0][-1]

        opp = game.players[1]
        opp_num_pieces = game.num_pieces[opp]

        player = game.players[0]
        player_num_pieces = game.num_pieces[player]

        if flip:
            opp_bar = features[0][292] * 2

            opp_off = int(np.floor(features[0][291] * opp_num_pieces))
            player_bar = features[0][145] * 2

            player_off = int(np.floor(features[0][144] * player_num_pieces))

            opp_checkers = features[0][147:291]
            player_checkers = features[0][0:144]

            # flip the view - this is just a perspective change and not an actual copy
            opp_checkers = opp_checkers[::-1]
            player_checkers = player_checkers[::-1]
        else:
            opp_bar = features[0][145] * 2
            opp_off = int(np.floor(features[0][144] * opp_num_pieces))
            player_bar = features[0][292] * 2
            player_off = int(np.floor(features[0][291] * player_num_pieces))

            opp_checkers = features[0][0:144]
            player_checkers = features[0][147:291]

        opp_max = np.argmax(opp_checkers == 1) // 6 if opp_bar == 0 else 0

        player_max = 23 - np.argmax(player_checkers[::-1] == 1) // 6 if player_bar == 0 else 23

        player_hit_count = 0
        opp_hit_count = 0

        for i in range(opp_max, player_max + 1):
            player_field_count = np.sum(player_checkers[i * 6:(i + 1) * 6])
            player_hit_count += 1 if player_field_count == 1 else 0
            opp_field_count = np.sum(opp_checkers[i * 6:(i + 1) * 6])
            opp_hit_count += 1 if opp_field_count == 1 else 0

        player_hit = player_hit_count / max(player_num_pieces - player_off - player_bar, 1)
        opp_hit = opp_hit_count / max(opp_num_pieces - opp_off - opp_bar, 1)

        if flip:
            hit = [[player_hit, opp_hit]]
        else:
            hit = [[opp_hit, player_hit]]

        features = np.append(features, hit, axis=1)

        return features

    def train(self, episodes=5000):
        self.mono_nn.create_model()

        # the agent plays against itself, making the best move for each player
        players = [TDAgent(Game.TOKENS[0], self), TDAgent(Game.TOKENS[1], self)]

        validation_interval = 1000

        for episode in range(episodes):
            # print()
            # print()
            # print('episode', episode)
            # if episode % validation_interval == 0:
            #     tester.test_self(self)
            #     tester.test_random(self)
                # self.print_checkpoints()

            game = Game.new()
            player_num = random.randint(0, 1)

            x = self.extract_features(game, players[player_num].player)

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

                x_next = self.extract_features(game, players[player_num].player)
                # print('next features extracted', x_next)
                V_next = 1 - self.mono_nn.get_output(x_next)
                # print('next output', V_next)

                self.mono_nn.run_output(x, V_next)

                x = x_next
                game_step += 1

            winner = game.winner()

            print("[Train %d/%d] (Winner: %s) in %d turns" % (episode, episodes, players[not winner].player, game_step))

            self.mono_nn.update_model(x, winner)

        self.mono_nn.training_end()

        tester.test_self(self)
