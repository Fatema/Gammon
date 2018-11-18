"""
This is a fork of https://github.com/awni/backgammon.
"""

from backgammon.game import Game

try:
    import pygame
    from pygame.locals import *
except:
    print("No module pygame, use command line to play")

import copy

OFF = Game.OFF
ON = Game.ON


class Agent:
    def __init__(self,player):
        self.player = player

    def get_action(self, moves, game=None):
        raise NotImplementedError("Override me")
