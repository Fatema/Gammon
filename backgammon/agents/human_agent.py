"""
This is a fork of https://github.com/awni/backgammon.
"""

import copy

import pygame

from backgammon.agents.agent import Agent
from backgammon.game import Game


class HumanAgent(Agent):
    def get_action(self, moves, game=None):
        loc = None
        moves_left = copy.deepcopy(moves)
        tmpg = game.clone()
        pmove = []
        while True:
            # if no more moves we break
            if not moves_left:
                break

            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    if loc is not None:
                        # check to see if we can move the piece
                        new_loc = game.grid_loc_from_pos(pos, self.player)
                        if new_loc is not None:
                            move = (loc,new_loc)
                            move_legit = False
                            new_moves = set()
                            for m in list(moves_left):
                                if m[0]==move:
                                    move_legit = True
                                    if m[1:]:
                                        new_moves.add(m[1:])
                            # if the move is legit we move it
                            if move_legit:
                                print(move)
                                pmove.append(move)
                                game.take_action((move,), self.player)
                                game.draw()
                                moves_left = new_moves
                                loc = None
                            else:
                                loc = new_loc
                    else:
                        # get a location to move
                        loc = game.grid_loc_from_pos(pos, self.player) # TODO implement this

    def get_action_command_line(self, moves, game=None):
        while True:
            if not moves:
                input("No moves for you...(hit enter)")
                break
            while True:
                mv1 = input("Please enter a move <location start,location end> ('%s' for off the board): " % Game.OFF)
                mv1 = self.get_formatted_move(mv1)
                if not mv1:
                    print('Bad format enter e.g. "3,4"')
                else:
                    break

            while True:
                mv2 = input("Please enter a second move (enter to skip): ")
                if mv2 == '':
                    mv2 = None
                    break
                mv2 = self.get_formatted_move(mv2)
                if not mv2:
                    print('Bad format enter e.g. "3,4"')
                else:
                    break

            if mv2:
                move = (mv1,mv2)
            else:
                move = (mv1,)

            if move in moves:
                break
            elif move[::-1] in moves:
                move = move[::-1]
                break
            else:
                print("You can't play that move")
        return move

    def get_formatted_move(self,move):
        try:
            start,end = move.split(",")
            if start != Game.ON:
                start = int(start)
            if end != Game.OFF:
                end = int(end)
            return (start,end)
        except:
            return False
