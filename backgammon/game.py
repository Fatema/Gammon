import os, sys
import numpy as np
import copy
try:
    import pygame
except:
    print("No module pygame, use screen drawing")


class Game:
    LAYOUT = "0-2-o,5-5-x,7-3-x,11-5-o,12-5-x,16-3-o,18-5-o,23-2-x"
    NUMCOLS = 24
    QUAD = 6
    OFF = 'off'
    ON = 'on'
    TOKENS = ['o','x']

    def __init__(self, layout=None, grid=None, off_pieces=None,
                 bar_pieces=None, num_pieces=None, gplayers=None):
        """
        Define a new game object
        """
        self.die = Game.QUAD
        self.layout = layout
        if grid:
            self.grid = copy.deepcopy(grid)
            self.off_pieces = copy.deepcopy(off_pieces)
            self.bar_pieces = copy.deepcopy(bar_pieces)
            self.num_pieces = copy.deepcopy(num_pieces)
            self.players = gplayers
            return
        self.players = Game.TOKENS
        self.grid = [[] for _ in range(Game.NUMCOLS)]
        self.off_pieces = {}
        self.bar_pieces = {}
        self.num_pieces = {}
        for t in self.players:
            self.bar_pieces[t] = []
            self.off_pieces[t] = []
            self.num_pieces[t] = 0
        self.init = True
        self.roll = None

    @staticmethod
    def new():
        game = Game()
        game.new_game()
        return game

    def clone(self):
        """
        Return an exact copy of the game. Changes can be made
        to the cloned version without affecting the original.
        """
        return Game(None, self.grid, self.off_pieces,
                    self.bar_pieces, self.num_pieces, self.players)
    
    def take_action(self, action, token):
        """
        Makes given move for player, assumes move is valid, 
        will remove pieces from play
        """
        ate_list = [0]*4
        for i,(s,e) in enumerate(action):
            if s == Game.ON:
                piece = self.bar_pieces[token].pop()
            else:
                piece = self.grid[s].pop()
            if e == Game.OFF:
                self.off_pieces[token].append(piece)
                continue
            if len(self.grid[e]) > 0 and self.grid[e][0] != token:
                bar_piece = self.grid[e].pop()
                self.bar_pieces[bar_piece].append(bar_piece)
                ate_list[i] = 1
            self.grid[e].append(piece)
        return ate_list

    def undo_action(self, action, player, ate_list):
        """
        Reverses given move for player, assumes move is valid, 
        will remove pieces from play
        """
        for i,(s,e) in enumerate(reversed(action)):
            if e == Game.OFF:
                piece = self.off_pieces[player].pop()
            else:
                piece = self.grid[e].pop()
                if ate_list[len(action) - 1 - i]:
                    bar_piece = self.bar_pieces[self.opponent(player)].pop()
                    self.grid[e].append(bar_piece)
            if s == Game.ON:
                self.bar_pieces[player].append(piece)
            else:
                self.grid[s].append(piece)

    def get_actions(self, roll, player, nodups=False):
        """
        Get set of all possible move tuples
        """
        moves = set()
        if nodups:
            start = 0
        else:
            start = None

        r1, r2 = roll
        if r1 == r2: # doubles
            i = 4
            # keep trying until we find some moves
            while not moves and i>0:
                self.find_moves(tuple([r1] * i), player, (), moves, start)
                i -= 1
        else:
            self.find_moves(roll, player, (), moves, start)
            self.find_moves((r2, r1), player, (), moves, start)
            # has no moves, try moving only one piece
            if not moves:
                for r in roll:
                    self.find_moves((r,), player, (), moves, start)

        return moves

    def find_moves(self, rs, player, move, moves, start=None):
        if len(rs)==0:
            moves.add(move)
            return
        r,rs = rs[0],rs[1:]
        # see if we can remove a piece from the bar
        if self.bar_pieces[player]:
            if self.can_onboard(player,r):
                piece = self.bar_pieces[player].pop()
                bar_piece = None
                if len(self.grid[r-1])==1 and self.grid[r-1][-1]!=player:
                    bar_piece = self.grid[r-1].pop()

                self.grid[r-1].append(piece)

                self.find_moves(rs, player, move + ((Game.ON, r - 1),), moves, start)
                self.grid[r-1].pop()
                self.bar_pieces[player].append(piece)
                if bar_piece:
                    self.grid[r-1].append(bar_piece)

            return

        # otherwise check each grid location for valid move using r
        offboarding = self.can_offboard(player)

        for i in range(len(self.grid)):
            if start is not None:
                start = i
            if self.is_valid_move(i,i+r,player):

                piece = self.grid[i].pop()
                bar_piece = None
                if len(self.grid[i+r]) == 1 and self.grid[i+r][-1]!=player:
                    bar_piece = self.grid[i+r].pop()
                self.grid[i+r].append(piece)
                self.find_moves(rs, player, move + ((i, i + r),), moves, start)
                self.grid[i+r].pop()
                self.grid[i].append(piece)
                if bar_piece:
                    self.grid[i+r].append(bar_piece)

            # If we can't move on the board can we take the piece off?
            if offboarding and self.remove_piece(player,i,r):
                piece = self.grid[i].pop()
                self.off_pieces[player].append(piece)
                self.find_moves(rs, player, move + ((i, Game.OFF),), moves, start)
                self.off_pieces[player].pop()
                self.grid[i].append(piece)

    def opponent(self,token):
        """
        Retrieve opponent players token for a given players token.
        """
        for t in self.players:
            if t!= token: return t

    def is_won(self, player):
        """
        If game is over and player won, return True, else return False
        """
        return self.is_over() and player == self.players[self.winner()]

    def is_lost(self, player):
        """
        If game is over and player lost, return True, else return False
        """
        return self.is_over() and player!=self.players[self.winner()]

    def reverse(self):
        """
        Reverses a game allowing it to be seen by the opponent
        from the same perspective
        """
        self.grid.reverse()
        self.players.reverse()

    def new_game(self):
        """
        Resets game to original layout.
        """
        for col in self.layout.split(','):
            loc,num,token = col.split('-')
            self.grid[int(loc)] = [token for _ in range(int(num))]
        for col in self.grid:
            for piece in col:
                self.num_pieces[piece] += 1

    def winner(self):
        """
        Get winner.
        """
        return len(self.off_pieces[self.players[0]])  ==  self.num_pieces[self.players[0]]

    def is_over(self):
        """
        Checks if the game is over.
        """
        for t in self.players:
            if len(self.off_pieces[t]) == self.num_pieces[t]:
                return True
        return False

    def can_offboard(self,player):
        count = 0
        for i in range(Game.NUMCOLS-self.die,Game.NUMCOLS):
            if len(self.grid[i]) > 0 and self.grid[i][0] == player:
                count += len(self.grid[i])
        if count+len(self.off_pieces[player]) == self.num_pieces[player]:
            return True
        return False

    def can_onboard(self,player,r):
        """
        Can we take a players piece on the bar to a position
        on the grid given by roll-1?
        """
        if len(self.grid[r-1]) <= 1 or self.grid[r-1][0] == player:
            return True
        else:
            return False

    def remove_piece(self,player,start,r):
        """
        Can we remove a piece from location start with roll r ?
        In this function we assume we are cool to offboard,
        i.e. no pieces on the bar and all are in the home quadrant.
        """
        if start < Game.NUMCOLS - self.die:
            return False
        if len(self.grid[start]) == 0 or self.grid[start][0] != player:
            return False
        if start+r  ==  Game.NUMCOLS:
            return True
        if start+r > Game.NUMCOLS:
            for i in range(start-1,Game.NUMCOLS-self.die-1,-1):
                if len(self.grid[i]) != 0 and self.grid[i][0] == self.players[0]:
                    return False
            return True
        return False

    def is_valid_move(self, start, end, token):
        if len(self.grid[start]) > 0 and self.grid[start][0] == token:
            if end < 0 or end >= len(self.grid):
                return False
            if len(self.grid[end]) <= 1:
                return True
            if len(self.grid[end]) > 1 and self.grid[end][-1] == token:
                return True
        return False

    def draw_col(self,i,col):
        print("|"),
        if i == -2:
            if col<10:
                print(""),
            print(str(col)),
        elif i == -1:
            print("--"),
        elif len(self.grid[col])>i:
            print(" ", self.grid[col][i]),
        else:
            print("  "),

    def draw_screen(self):
        os.system('clear')
        half_grid_size = int(len(self.grid)/2)
        grid_size = len(self.grid)
        largest = max([len(self.grid[i]) for i in range(half_grid_size, grid_size)])
        for i in range(-2,largest):
            for col in range(half_grid_size,len(self.grid)):
                self.draw_col(i,col)
            print("|")
        print()
        print()
        largest = max([len(self.grid[i]) for i in range(half_grid_size)])
        for i in range(largest-1,-3,-1):
            for col in range(half_grid_size-1,-1,-1):
                self.draw_col(i,col)
            print("|")
        for t in self.players:
            print("<Player %s>  Off Board : "%(t)),
            for piece in self.off_pieces[t]:
                print(t+''),
            print("   Bar : "),
            for piece in self.bar_pieces[t]:
                print(t+''),
            print()

    def draw(self,roll=None):
        if roll is None:
            roll = self.roll
        else:
            self.roll = roll

        self.draw_gui(roll)

    def init_gui(self):
        pygame.init()
        WIDTH = 800
        HEIGHT = 425
        
        WOFFSET_TOP = 57
        HOFFSET_TOP = 12
        HOFFSET_BOT = 370
        WSKIP = 55
        WMID = 32

        HSKIP = 30
        size = WIDTH,HEIGHT

        self.grid_locs = []

        for i in range(24):
            mid = 0
            hoff = HOFFSET_TOP
            hskip = HSKIP
            k = 11-i
            if i < 6 or i>17:
                mid = WMID
            if i > 11:
                hoff = HOFFSET_BOT
                hskip = -hskip
                k = i-12
            self.grid_locs.append([(WOFFSET_TOP + k * WSKIP + mid, hoff + j * hskip) for j in range(6)])

        self.bar_locs = {'x':[(376, 142), (376, 110)], 'o':[(376, 243), (376, 275)]}
        self.board_img = pygame.transform.scale(pygame.image.load('images/board.png'),size)
        self.screen = pygame.display.set_mode(self.board_img.get_rect().size)
        self.tokIms = {'x':pygame.image.load('images/whitePiece.png'),
                           'o':pygame.image.load('images/blackPiece.png')}
        self.dies = [pygame.transform.scale(pygame.image.load('images/die%d.png'%i),(35,35))
                         for i in range(1,7)]
        self.offIms = {'x':pygame.transform.scale(pygame.image.load('images/whiteOff.png'),(40,18)),
                           'o':pygame.transform.scale(pygame.image.load('images/blackOff.png'),(40,18))}
        
        outOff = 748
        bOffH = 391
        wOffH = 11
        offSkip = 9
        self.offLocs = {'x':[(outOff,bOffH-i*offSkip) for i in range(19)],
                            'o':[(outOff,wOffH+i*offSkip) for i in range(19)]}
        
    def draw_gui(self, roll):
        if self.init:
            self.init_gui()
        self.screen.blit(self.board_img,self.board_img.get_rect())
        self.screen.blit(self.dies[roll[0]-1],(180,190))
        self.screen.blit(self.dies[roll[1]-1],(220,190))
        for i,col in enumerate(self.grid):
            for j,t in enumerate(col):
                # for now only draw first 6 pieces
                if j > 5:
                    break
                self.screen.blit(self.tokIms[t], self.grid_locs[23 - i][j])
        for k,v in self.bar_pieces.items():
            for i,t in enumerate(v):
                if i>1:
                    break
                self.screen.blit(self.tokIms[t], self.bar_locs[t][i])

        for k,v in self.off_pieces.items():
            for i,t in enumerate(v):
                self.screen.blit(self.offIms[t],self.offLocs[t][i])
        pygame.display.flip()

    def grid_loc_from_pos(self, pos, player):
        tx,ty = self.tokIms['x'].get_rect().size

        def on_piece(piece_loc, pos, sizex, sizey):
            px,py = piece_loc
            tx,ty = pos
            if tx < px + sizex and tx > px:
                if ty < py + sizey and ty > py:
                    return True
            return False
            
        # find out if we are on the grid
        for i,col in enumerate(self.grid):
            for loc in self.grid_locs[23 - i]:
                if on_piece(loc,pos,tx,ty):
                    return i

        # find out if we are on the bar
        for i,bp in enumerate(self.bar_pieces[player]):
            if on_piece(self.bar_locs[player][i], pos, tx, ty):
                return Game.ON

        # find out if we are removing pieces
        offBase = self.offLocs['o'][0] if player == 'o' else self.offLocs['x'][-1]
        offHeight = 200
        offWidth,_ = self.offIms['x'].get_rect().size
        if on_piece(offBase,pos, offWidth, offHeight):
            return Game.OFF
        
        return None

    def extract_features(self, player):
        features = []
        for p in self.players:
            for col in self.grid:
                feats = [0.] * 6
                if len(col) > 0 and col[0] == p:
                    for i in range(len(col)):
                        feats[min(i, 5)] += 1
                features += feats
            features.append(float(len(self.bar_pieces[p])) / 2.)
            features.append(float(len(self.off_pieces[p])) / self.num_pieces[p])
        if player == self.players[0]:
            features += [1., 0.]
        else:
            features += [0., 1.]
        return np.array(features).reshape(1, -1)


if __name__ == '__main__':
    g = Game(Game.LAYOUT)
    g.new_game()
    g.draw((4,3))
