import Othello as o
import random
import math
import torch

class NeuralMonteCarlo:

    def __init__(self):
        pass

# Convert board state into a tensor of size 1 x 3 x 8 x 8
def board_state_to_tensor(white, black, turn_count):
    if turn_count % 2 == 0:
        player, opponent = black, white
    else:
        player, opponent = white, black

    possible_moves = o.advanced_gen_moves(player, opponent)

    t = torch.zeros((1,3,8,8))

    for i in range(8):
        for j in range(8):
            if o.read_bit(player, i*8+j):
                t[0,0,i,j] = 1
            elif o.read_bit(opponent, i*8+j):
                t[0,1,i,j] = 1
            elif o.read_bit(possible_moves, i*8+j):
                t[0,2,i,j] = 1

    return t

class Node:

    def __init__(self, network, parent=None, white=34628173824, black=68853694464, turn_count=0, move_to_reach=-1):
        # Properties relevant to a node of a monte carlo tree
        self.parent = parent
        self.children = []

        self.visits = 0
        self.score = 0

        # Properties of the current Othello game state
        self.white = white
        self.black = black
        self.turn_count = turn_count
        self.move_to_reach = move_to_reach # What move created this Node?

        # Note: When true, black is to play. When false, white is to play
        self.to_play = turn_count % 2 == 0

        if self.to_play:
            self.available_moves = o.get_valid_move_list(black, white)
        else:
            self.available_moves = o.get_valid_move_list(white, black)

        if len(self.available_moves) == 0:
            self.available_moves.append(-1)  # Pass token

        # Get predictions from the network
        p, v = network(board_state_to_tensor(white, black, turn_count))

        # Backpropagate value through the tree
        value = v[0].item()
        self.backpropogate(value)

        # Translate p to only include probabilities for valid moves
        mask = [False] * 65
        for move in self.available_moves:
            if move != -1:
                mask[move] = True
            else:
                mask[64] = True
        mask_t = torch.tensor(mask, dtype=torch.bool)

        p[~mask_t] = -1e9 # Mask illegal moves with large negative coefficient
        dist = torch.softmax(p, dim=0)

        # Look-up table for the probability to make a certain action, based on the network.
        self.probabilities = {}
        for i in range(len(self.available_moves)):
            move = self.available_moves[i]

            if move != -1:
                self.probabilities[move] = dist[i].item()
            else:
                self.probabilities[move] = dist[64].item()



    def is_explored(self):
        return len(self.available_moves) == 0

    def make_child(self, move = None):
        if move is None: move = random.choice(self.available_moves)

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

    def backpropogate(self, value):
        self.score += value
        self.visits += 1

        if self.parent is not None:
            self.parent.backpropogate(-value)