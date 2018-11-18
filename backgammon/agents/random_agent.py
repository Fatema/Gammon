"""
This is a fork of https://github.com/awni/backgammon.
"""

import random

from backgammon.agents.agent import Agent


class RandomAgent(Agent):
    def get_action(self, moves, game=None):
        if moves:
            return random.choice(list(moves))
        return None
