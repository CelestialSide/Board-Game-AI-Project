from win32gui import UpdateLayeredWindow

import Othello as o
import random
import math

class Node:

    def __init__(self, parent=None, white=34628173824, black=68853694464, turn_count=0):
        # Properties relevant to a node of a monte carlo tree
        self.parent = parent
        self.children = []

        self.visits = 0
        self.score = 0

        # Properties of the current Othello game state
        self.white = white
        self.black = black
        self.turn_count = turn_count

        # Note: When true, black is to play. When false, white is to play
        self.to_play = turn_count % 2 == 0

        if self.to_play:
            self.available_moves = o.get_valid_move_list(black, white)
        else:
            self.available_moves = o.get_valid_move_list(white, black)

        if len(self.available_moves) == 0:
            self.available_moves.append(-1) # Pass token

    def is_explored(self):
        return len(self.available_moves) == 0

    def make_child(self):
        move = random.choice(self.available_moves)

        if move != -1:
            if self.to_play:
                new_black, new_white = o.update_board(move, self.black, self.white)
            else:
                new_white, new_black = o.update_board(move, self.white, self.black)
        else:
            # Pass -> boards unchanged
            new_black, new_white = self.white, self.black

        self.available_moves.remove(move)

        child = Node(self, new_white, new_black, self.turn_count + 1)
        self.children.append(child)

        return child

    def compute_UCT(self, c):
        return self.score / self.visits + c * math.sqrt(math.log(self.parent.visits)/self.visits)