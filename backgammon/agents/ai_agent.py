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
        v_best = 0
        a_best = None

        """
        1-ply method, it is greedy selection for an action
        """
        # print('player', self.player,'considering actions:', actions)
        for a in actions:
            # print('action considered:', a)
            ateList = game.take_action(a, self.player)
            # print('action taken outcome:', ateList)
            # I can change this so it uses the model extract features
            features = self.model.extract_features(game, game.opponent(self.player))
            # game.draw_screen()
            _, v = self.model.get_output(features)
            # print('NN output', v)
            v_w = 1 - v[0][0] if self.player == game.players[0] else v[0][0]
            # v_w = v[0][0]
            # v_w = 2 * v[0][0] - 1 + 2 * (v[0][1] - v[0][2])
            if v_w > v_best:
                v_best = v_w
                a_best = a
            game.undo_action(a, self.player, ateList)

        # print('features after action:', features)

        # print('best action selected', a_best, v_best, self.player)

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
