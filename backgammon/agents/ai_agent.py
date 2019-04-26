"""
This is a fork of https://github.com/awni/backgammon.
"""

from backgammon.agents import agent
import numpy as np

class TDAgent:
    def __init__(self, player, model):
        self.player = player
        self.model = model
        self.name = 'TD-Gammon'

    def get_action(self, actions, game=None):
        """
        Return best action according to self.evaluationFunction,
        with no lookahead.
        """
        v_best = -1 # game got stuck when this was 0, wasn't happy with games being stuck
        a_best = None

        """
        1-ply method, it is greedy selection for an action
        """
        # print('considering actions:', actions)
        for a in actions:
            # print('action considered:', a)
            ateList = game.take_action(a, self.player)
            # print('action taken outcome:', ateList)
            # I can change this so it uses the model extract features
            features = self.model.extract_features(game, game.opponent(self.player))
            # game.draw_screen()
            # print('features after action:', features)
            _, v = self.model.get_output(features)
            # print(v)
            # print('NN output', v)
            if v > v_best:
                v_best = v
                a_best = a
            game.undo_action(a, self.player, ateList)

        # print('best action selected', a_best, v_out, self.player)

        return a_best


class PubevalAgent:
    def __init__(self, player):
        self.player = player
        self.name = 'pubeval'

    wc = np.array([0.25696, -0.66937, -1.66135, -2.02487, -2.53398, -0.16092, -1.11725, -1.06654,
                   -0.92830, -1.99558, -1.10388, -0.80802, 0.09856, -0.62086, -1.27999, -0.59220,
                   -0.73667, 0.89032, -0.38933, -1.59847, -1.50197, -0.60966, 1.56166, -0.47389,
                   -1.80390, -0.83425, -0.97741, -1.41371, 0.24500, 0.10970, -1.36476, -1.05572,
                   1.15420, 0.11069, -0.38319, -0.74816, -0.59244, 0.81116, -0.39511, 0.11424,
                   -0.73169, -0.56074, 1.09792, 0.15977, 0.13786, -1.18435, -0.43363, 1.06169,
                   -0.21329, 0.04798, -0.94373, -0.22982, 1.22737, -0.13099, -0.06295, -0.75882,
                   -0.13658, 1.78389, 0.30416, 0.36797, -0.69851, 0.13003, 1.23070, 0.40868,
                   -0.21081, -0.64073, 0.31061, 1.59554, 0.65718, 0.25429, -0.80789, 0.08240,
                   1.78964, 0.54304, 0.41174, -1.06161, 0.07851, 2.01451, 0.49786, 0.91936,
                   -0.90750, 0.05941, 1.83120, 0.58722, 1.28777, -0.83711, -0.33248, 2.64983,
                   0.52698, 0.82132, -0.58897, -1.18223, 3.35809, 0.62017, 0.57353, -0.07276,
                   -0.36214, 4.37655, 0.45481, 0.21746, 0.10504, -0.61977, 3.54001, 0.04612,
                   -0.18108, 0.63211, -0.87046, 2.47673, -0.48016, -1.27157, 0.86505, -1.11342,
                   1.24612, -0.82385, -2.77082, 1.23606, -1.59529, 0.10438, -1.30206, -4.11520,
                   5.62596, -2.75800])

    wr = np.array([0.00000, -0.17160, 0.27010, 0.29906, -0.08471, 0.00000, -1.40375, -1.05121,
                   0.07217, -0.01351, 0.00000, -1.29506, -2.16183, 0.13246, -1.03508, 0.00000,
                   -2.29847, -2.34631, 0.17253, 0.08302, 0.00000, -1.27266, -2.87401, -0.07456,
                   -0.34240, 0.00000, -1.34640, -2.46556, -0.13022, -0.01591, 0.00000, 0.27448,
                   0.60015, 0.48302, 0.25236, 0.00000, 0.39521, 0.68178, 0.05281, 0.09266,
                   0.00000, 0.24855, -0.06844, -0.37646, 0.05685, 0.00000, 0.17405, 0.00430,
                   0.74427, 0.00576, 0.00000, 0.12392, 0.31202, -0.91035, -0.16270, 0.00000,
                   0.01418, -0.10839, -0.02781, -0.88035, 0.00000, 1.07274, 2.00366, 1.16242,
                   0.22520, 0.00000, 0.85631, 1.06349, 1.49549, 0.18966, 0.00000, 0.37183,
                   -0.50352, -0.14818, 0.12039, 0.00000, 0.13681, 0.13978, 1.11245, -0.12707,
                   0.00000, -0.22082, 0.20178, -0.06285, -0.52728, 0.00000, -0.13597, -0.19412,
                   -0.09308, -1.26062, 0.00000, 3.05454, 5.16874, 1.50680, 5.35000, 0.00000,
                   2.19605, 3.85390, 0.88296, 2.30052, 0.00000, 0.92321, 1.08744, -0.11696,
                   -0.78560, 0.00000, -0.09795, -0.83050, -1.09167, -4.94251, 0.00000, -1.00316,
                   -3.66465, -2.56906, -9.67677, 0.00000, -2.77982, -7.26713, -3.40177, -12.32252,
                   0.00000, 3.42040])

    def pubeval(self, game):
        pos, race = self.board2pos(game)

        # all men off, best possible move
        if pos[26] == 15:
            return 99999999.

        x = self.setx(pos)  # sets input array x[]

        if race:  # use race weights
            score = np.sum(self.wr * x)
        else:  # use contact weights
            score = np.sum(self.wc * x)

        return score

    def setx(self, pos):
        x = [0.0 for _ in range(120)]

        # first encode board locations 24-1
        for j in range(1, 25):
            jm1 = j - 1
            n = pos[25 - j]
            if n != 0:
                if n == -1: x[5 * jm1 + 0] = 1.0
                if n == 1: x[5 * jm1 + 1] = 1.0
                if n >= 2: x[5 * jm1 + 2] = 1.0
                if n == 3: x[5 * jm1 + 3] = 1.0
                if n >= 4: x[5 * jm1 + 4] = (n - 3) / 2.0

        # encode opponent barmen
        x += [pos[0] / -2.0]

        # encode computer's menoff
        x += [pos[26] / 15.0]

        return np.array(x)

    def board2pos(self, game):
        pos = [0 for _ in range(28)]
        for k in range(len(game.players)):
            p = game.players[k]
            for j in range(len(game.grid)):
                col = game.grid[j]
                if len(col) > 0 and col[0] == p:
                    if k == 0:
                        pos[24 - j] = len(col)
                    else:
                        pos[24 - j] = len(col) * -1
        pos[0] = -len(game.bar_pieces[game.players[1]])
        pos[25] = len(game.bar_pieces[game.players[0]])
        pos[26] = len(game.off_pieces[game.players[0]])
        pos[27] = -len(game.off_pieces[game.players[1]])

        i = 0
        # find location for last opponent checker looking left to right
        while i < 24:
            if pos[i] < 0:
                break
            i += 1

        j = 23
        # find location for last player checker looking right to left
        while j > 0:
            if pos[j] > 0:
                break
            j -= 1

        race = i > j # all checkers are past each other

        return pos, race

    def get_action(self, actions, game=None):
        """
        Return best action according to pubeval function
        """
        score_best = np.finfo('float32').min
        a_best = None

        for a in actions:
            ateList = game.take_action(a, self.player)

            score = self.pubeval(game)

            if score > score_best:
                score_best = score
                a_best = a
            game.undo_action(a, self.player, ateList)

        # print('best action selected', a_best, v_out, self.player)

        return a_best


def nnetEval(game, player, weights):
    w1, w2, b1, b2 = weights
    features = np.array(game.extract_features(player)).reshape(-1, 1)
    hiddenAct = 1 / (1 + np.exp(-(w1.dot(features) + b1)))
    v = 1 / (1 + np.exp(-(w2.dot(hiddenAct) + b2)))
    return v


class ExpectiMiniMaxAgent(agent.Agent, object):
    def miniMaxNode(self, game, player, roll, depth):
        actions = game.get_actions(roll, player, nodups=True)
        rollScores = []

        if player == self.player:
            scoreFn = max
        else:
            scoreFn = min
            depth -= 1

        if not actions:
            return self.expectiNode(game, game.opponent(player), depth)
        for a in actions:
            ateList = game.take_action(a, player)
            rollScores.append(self.expectiNode(game, game.opponent(player), depth))
            game.undo_action(a, player, ateList)

        return scoreFn(rollScores)

    def expectiNode(self, game, player, depth):
        if depth == 0:
            return self.evaluationFunction((game, player), self.evaluationArgs)

        total = 0
        for i in range(1, game.die + 1):
            for j in range(i + 1, game.die + 1):
                score = self.miniMaxNode(game, player, (i, j), depth)
                if i == j:
                    total += score
                else:
                    total += 2 * score

        return total / float(game.die ** 2)

    def get_action(self, actions, game=None):
        depth = 1
        if len(actions) > 100:
            depth = 0
        outcomes = []
        for a in actions:
            ateList = game.take_action(a, self.player)
            score = self.expectiNode(game, game.opponent(self.player), depth)
            game.undo_action(a, self.player, ateList)
            outcomes.append((score, a))
        action = max(outcomes)[1]
        return action

    def __init__(self, player, evalFn, evalArgs=None):
        super(self.__class__, self).__init__(player)
        self.evaluationFunction = evalFn
        self.evaluationArgs = evalArgs
