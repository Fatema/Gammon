from __future__ import division

import random

import tester
from backgammon.agents.ai_agent import TDAgent
from backgammon.agents.human_agent import HumanAgent
from backgammon.game import Game

from subnet import *

class Modnet:
    def __init__(self, model_path, summary_path, checkpoint_path, restore=False):
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
        self.restore = restore

        self.default_net = self.create_network('default')

        self.racing_net = self.create_network('racing')

        self.holding_net = self.create_network('holding')

        self.networks = {'d' : self.default_net, 'r' : self.racing_net, 'h' : self.holding_net}

    def create_network(self, name):
        network = SubNet()
        network.set_network_name(name)
        network.set_paths(self.model_path, self.summary_path, self.checkpoint_path)
        network.start_session(restore=self.restore)
        return network

    def set_previous_checkpoint(self):
        for net in self.networks:
            self.networks[net].set_previous_checkpoint()

    def print_checkpoints(self):
        for net in self.networks:
            self.networks[net].print_checkpoints()

    def restore_previous(self):
        for net in self.networks:
            self.networks[net].restore_previous()

    # this method is not really related to the model but it is encapsulated as part of the model class
    def play(self):
        game = Game.new()
        game.play([HumanAgent(Game.TOKENS[0]), TDAgent(Game.TOKENS[1], self)], draw=True)

    # gating program - decide which subnet to run based on the input features
    def get_output(self, x):
        # make the indexes evaulated based on global variables
        flip = x[0][-1]

        if flip:
           opp_pip = x[0][292]
           player_pip = x[0][145]

           opp_checkers = x[0][147:291]
           player_checkers = x[0][0:144]

           # flip the view - this is just a perspective change and not an actual copy
           opp_checkers = opp_checkers[::-1]
           player_checkers = player_checkers[::-1]
        else:
            opp_pip = x[0][145]
            player_pip = x[0][292]

            opp_checkers = x[0][0:144]
            player_checkers = x[0][147:291]

        # the calculation is based on the player view
        # min means that it is closer to the perspective player home and max is the opposite
        opp_min = 23 - np.argmax(opp_checkers[::-1]) // 6
        opp_max = np.argmax(opp_checkers) // 6

        player_min = np.argmax(player_checkers) // 6
        player_max = 23 - np.argmax(player_checkers[::-1]) // 6

        net = 'd'

        if player_max < opp_max or player_pip > opp_pip:
            net = 'r'
        else:
            player_close_pos = 0
            player_trapped_pos = 0
            player_trapped_count = 0
            opp_trapped_pos = 0
            opp_trapped_count = 0

            for i in range(player_min, opp_max):
                player_close_pos += player_checkers[i * 6]

            for i in range(opp_max + 1, player_max + 1):
                # if sum is 1 move to a defensive strategy
                sum = np.sum(player_checkers[i * 6:(i + 1) * 6])
                player_trapped_pos += 1 if sum > 0 else 0
                player_trapped_count += sum

            if player_trapped_pos < 3 and player_trapped_count < 4:
                net = 'r'

        return net, self.networks[net].get_output(x)

    def extract_features(self, game, player):
        features = []
        # print(player)
        # the order in which the players are evaluated matters
        for k in range(len(game.players)):
            p = game.players[k]
            pip_count = 0
            for j in range(len(game.grid)):
                col = game.grid[j]
                feats = [0.] * 4
                if len(col) > 0 and col[0] == p:
                    if k == 0:
                        temp = len(col) * (24 - j)
                        pip_count += temp
                        # print(p,'count per col', j, temp, pip_count, len(col))
                    else:
                        temp = len(col) * (j + 1)
                        pip_count += temp
                        # print(p,'count per col', j, temp, pip_count, len(col))
                    # set the features to be 4 units each, last unit is set to (n-3)/2
                    for i in range(len(col)):
                        if i >= 3: break
                        feats[i] += 1
                    feats[3] = (len(col) - 3) / 2. if len(col) > 3 else 0
                features += feats
            # print('pip_count before off pieces', pip_count)
            features.append(float(len(game.bar_pieces[p])) / 2.) # td gammon had it like this to scale the range between 0 and 1
            features.append(float(len(game.off_pieces[p])) / game.num_pieces[p])
            # pip_count for the player the closer to home the less the value is
            # print(game.bar_pieces[p], game.off_pieces[p])
            pip_count += len(game.bar_pieces[p]) * 24
            features.append(float(pip_count))
            # print('pip count for', p, pip_count)
        if player == game.players[0]:
            features += [1., 0.]
        else:
            features += [0., 1.]
        return np.array(features).reshape(1, -1)

    def train(self, episodes=5000):
        for net in self.networks:
            self.networks[net].create_model()

        # the agent plays against itself, making the best move for each player
        players = [TDAgent(Game.TOKENS[0], self), TDAgent(Game.TOKENS[1], self)]

        validation_interval = 1000

        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0:
                tester.test_self(self)
                # self.print_checkpoints()

            game = Game.new()
            game.generate_random_game()
            player_num = random.randint(0, 1)

            x = self.extract_features(game, players[player_num].player)

            game_step = 0
            while not game.is_over():
                roll = game.roll_dice()
                if player_num:
                    game.reverse()
                game.take_turn(players[player_num], roll, nodups=True)
                if player_num:
                    game.reverse()
                player_num = (player_num + 1) % 2

                x_next = self.extract_features(game, players[player_num].player)
                gated_net, V_next = self.get_output(x_next)

                self.networks[gated_net].run_output(x, V_next)

                x = x_next
                game_step += 1

            winner = game.winner()

            for net in self.networks:
                self.networks[net].update_model(x, winner, episode, episodes, players, game_step)

        for net in self.networks:
            self.networks[net].training_end()

        tester.test_self(self)
