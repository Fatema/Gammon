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

        self.priming_net = self.create_network('priming')

        self.backgame_net = self.create_network('backgame')

        self.networks = {'d': self.default_net,
                         'r': self.racing_net,
                         'p': self.priming_net,
                         'b': self.backgame_net}

    def create_network(self, name):
        network = SubNet()
        network.set_network_name(name)
        network.set_paths(self.model_path, self.summary_path, self.checkpoint_path)
        network.start_session(restore=self.restore)
        return network

    def print_checkpoints(self):
        for net in self.networks:
            self.networks[net].print_checkpoints()

    def print_checkpoints(self):
        for net in self.networks:
            self.networks[net].print_checkpoints()

    def restore_previous(self):
        for net in self.networks:
            self.networks[net].restore_previous()

    def restore_test_checkpoint(self, timestamp, game_number):
        for net in self.networks:
            self.networks[net].restore_test_checkpoint(timestamp, game_number)

    # this method is not really related to the model but it is encapsulated as part of the model class
    def play(self):
        game = Game.new()
        game.play([HumanAgent(Game.TOKENS[0]), TDAgent(Game.TOKENS[1], self)], draw=True)

    # gating program - decide which subnet to run based on the input features
    def get_output(self, x):
        # make the indexes evaulated based on global variables
        flip = x[0][-1]

        if flip:
           opp_pip = x[0][293]
           opp_bar = x[0][292] * 2
           player_pip = x[0][146]
           player_bar = x[0][145] * 2

           opp_checkers = x[0][147:291]
           player_checkers = x[0][0:144]

           # flip the view - this is just a perspective change and not an actual copy
           opp_checkers = opp_checkers[::-1]
           player_checkers = player_checkers[::-1]
        else:
            opp_pip = x[0][146]
            opp_bar = x[0][145] * 2
            player_pip = x[0][293]
            player_bar = x[0][292] * 2

            opp_checkers = x[0][0:144]
            player_checkers = x[0][147:291]

        # the calculation is based on the player view
        # min means that it is closer to the perspective player home and max is the opposite
        # opp_min = 23 - np.argmax(opp_checkers[::-1]) // 6
        opp_max = np.argmax(opp_checkers) // 6

        # player_min = np.argmax(player_checkers) // 6
        player_max = 23 - np.argmax(player_checkers[::-1]) // 6

        net = 'd'

        if player_max < opp_max and opp_bar == 0 and player_bar == 0:
            net = 'r'
        else:
            # player_close_pos = 0
            player_trapped_pos = 0
            player_trapped_count = 0
            player_prime = [0]
            j = 0

            # for i in range(player_min, opp_max):
            #     player_close_pos += player_checkers[i * 6]

            for i in range(opp_max + 1, player_max + 1):
                # if sum is 1 move to a defensive strategy
                sum = np.sum(player_checkers[i * 4:(i + 1) * 4])
                player_trapped_pos += 1 if sum > 0 else 0
                player_trapped_count += sum
                if sum > 1:
                    player_prime[j] += 1
                else:
                    player_prime += [0]
                    j += 1

                    # check for prime then do priming game
                    if max(player_prime) > 4:
                        net = 'p'
                    # check if the player is at a disadvantage and check for the checkers at opponent home if they
                    # are more than 3 along with the checkers on the bar do the back game
                    elif player_pip - opp_pip > 90 and np.sum(opp_checkers[108:144]) + player_bar > 3:
                        net = 'b'

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
                        feats[min(i, 5)] += 1
                features += feats
            # print('pip_count before off pieces', pip_count)
            features.append(float(len(game.bar_pieces[p])) / 2.) # td gammon had it like this to scale the range between 0 and 1
            features.append(float(len(game.off_pieces[p])) / game.num_pieces[p])
            # pip_count for the player the closer to home the less the value is
            # print(game.bar_pieces[p], game.off_pieces[p])
            pip_count += len(game.bar_pieces[p]) * 25
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
            if episode % validation_interval == 0:
                tester.test_self(self)
                tester.test_random(self)
                # self.print_checkpoints()

            game = Game.new()
            # game.generate_random_game()
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

            print("[Train %d/%d] (Winner: %s) in %d turns" % (episode, episodes, players[not winner].player, game_step))

            for net in self.networks:
                self.networks[net].update_model(x, winner)

        for net in self.networks:
            self.networks[net].training_end()

        tester.test_self(self)
