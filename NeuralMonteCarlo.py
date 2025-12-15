import Othello as o
import random
import math
import torch
from AlphaZeroNetwork import AlphaZeroNet
import numpy as np

class NeuralMonteCarlo:

    def __init__(self, network, white=68853694464, black=34628173824, turn_count=0, move_to_reach=-1, train_mode=False):
        self.root = Node(network, None, white, black, turn_count, move_to_reach, use_dir_noise=train_mode, is_pending=False)
        self.network = network
        self.training = train_mode

        self.c_puct = 1 # Hyperparameter constant that affects exploration rate.

    def shift_root(self, white, black, turn_count, move):
        for child in self.root.children:
            if child.move_to_reach == move:
                assert not child.is_pending, "Child was not evaluated!"

                child.parent = None
                self.root = child

                if self.training:
                    self.root.add_noise()

                return
        
        # If move is not in tree, create a new root
        self.root = Node(self.network, None, white, black, turn_count, move, use_dir_noise=self.training, is_pending=False)

    def run_simulation(self, batch_size=1):
        """
        Traverse the tree, guided by the PUCT score, and create a new node in the tree.
        The tree is updated with the new simulation.
        :return: None
        """
        pending = []
        
        for _ in range(batch_size):

            current_node = self.root

            # Traverse the tree a record a new node.
            while True:
                # If the game is over in the current state, don't make more children. Get value from direct
                # board result
                if o.is_game_over(current_node.white, current_node.black):
                    winner = o.determine_winner(current_node.black, current_node.white) # 1 - Black, 0 - Draw, -1 - White
                    if current_node.turn_count % 2 == 1:
                        winner *= -1

                    current_node.backpropogate(winner)

                    # Since we're done searching, break the traversal
                    break
                
                # If this node still needs children, append a child
                if not current_node.is_explored():
                    child = current_node.make_child(self.network)
                    child.is_pending = True
                    pending.append(child)
                    break
                
                # If node is explored, go to its best child.
                available_children = self.get_selectable(current_node)
                if len(available_children) == 0:
                    break
                    
                selected_child = 0
                highest_PUCT = available_children[0].determine_PUCT(self.c_puct)

                for i in range(1, len(available_children)):
                    puct = available_children[i].determine_PUCT(self.c_puct)

                    if puct > highest_PUCT:
                        selected_child = i
                        highest_PUCT = puct

                current_node = available_children[selected_child]
        
        if len(pending) == 0:
            return
        
        # Evaluate all of the pending nodes & backpropogate
        states = torch.cat([ board_state_to_tensor(n.white, n.black, n.turn_count) for n in pending ], dim=0).to(next(self.network.parameters()).device)

        with torch.no_grad():
            ps, vs = self.network(states)
        
        for i, node in enumerate(pending):
            node.evaluate(ps[i], vs[i])


    
    def get_selectable(self, node):
        return [c for c in node.children if not c.is_pending]

    def run_iterations(self, num_its, mcts_batch_size=8):
        for i in range(num_its // mcts_batch_size + 1):
            self.run_simulation(mcts_batch_size)

    def get_root_visit_distribution(self):
        """
        Return the probability distribution corresponding to how often
        every child of the root was visited.
        [DEPRECATED - returns numpy array instead, 64=pass] :return: A dictionary of (move, probability) pairs. -1 (pass) is moved to token 64
        """
        # N = self.root.visits
        N = sum([child.visits for child in self.root.children])
        dist = np.zeros(65)

        if len(self.root.children) == 0:
            dist[64] = 1.0
        elif N > 0:
            for child in self.root.children:
                child_move = child.move_to_reach

                if child_move != -1:
                    dist[child_move] = child.visits / N
                else:
                    dist[64] = child.visits / N
        else:
            # Uniform fallback
            N = len(self.root.move_options)

            for m in self.root.move_options:
                if m != -1:
                    dist[m] = 1 / N
                else:
                    dist[64] = 1 / N

        return dist

    def get_move_to_play(self, temperature=0.0):
        """
        Get the move that should be played from the root's state
        :return:
        """

        # If no children, we must pass (terminal node)
        if len(self.root.children) == 0:
            return -1

        # The best move has the most visits
        chosen_dex = -1
        visit_range = torch.tensor([child.visits for child in self.root.children], dtype=torch.float32)

        if temperature > 0:
            # Apply temperature
            probs = visit_range.pow(1.0 / temperature)

            # Sample uniformly if children havent been visited.
            if probs.sum() == 0.0:
                probs = torch.ones_like(visit_range)

            probs = probs / probs.sum()

            # Sample from distribution
            chosen_dex = torch.multinomial(probs, num_samples=1).item()
        else:
            chosen_dex = torch.argmax(visit_range).item()

        return self.root.children[chosen_dex].move_to_reach


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

    def __init__(self, network, parent=None, white=68853694464, black=34628173824, turn_count=0, move_to_reach=-1, use_dir_noise=False, is_pending=True):
        # Properties relevant to a node of a monte carlo tree
        self.parent = parent
        self.children = []
        self.use_dir_noise = use_dir_noise
        self.is_pending = is_pending

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
        
        self.move_options = self.available_moves.copy()

        # ---------------------- Built-in Backpropagation (Since we need to evaluate probabilities)
        # Get predictions from the network
        if not is_pending:
            p, v = network(board_state_to_tensor(white, black, turn_count).to(next(network.parameters()).device))
            p = p.cpu()
            v = v.cpu()

            self.evaluate(p[0], v[0])
        
        if use_dir_noise:
            self.add_noise()

        

    def evaluate(self, p, v):
        # P Shape: (65,)
        # V Shape: (1,)

        # Backpropagate value through the tree
        if self.parent is not None:
            self.parent.backpropogate(v.item()) # We start with parent since we haven't visited this node - only created it.

        # Translate p to only include probabilities for valid moves
        mask = [False] * 65
        for move in self.available_moves:
            if move != -1:
                mask[move] = True
            else:
                mask[64] = True
        mask_t = torch.tensor(mask, dtype=torch.bool)

        p[~mask_t] = -1e9 # Mask illegal moves with large negative coefficient. Goes to 0 in softmax.
        dist = torch.softmax(p, dim=0)

        # Look-up table for the probability to make a certain action, based on the network.
        self.probabilities = {}
        for i in range(len(self.available_moves)):
            move = self.available_moves[i]

            if move != -1:
                self.probabilities[move] = dist[move].item()
            else:
                self.probabilities[move] = dist[64].item()
        
        self.is_pending = False

    def add_noise(self):
        # Add a bit of noise to this distribution if this is a root
        if self.use_dir_noise:
            noise = np.random.dirichlet([0.25] * len(self.available_moves))
            for i, move in enumerate(self.available_moves):
                self.probabilities[move] = (
                    (1 - 0.25) * self.probabilities[move] + 0.25 * noise[i]
                )

    def is_explored(self):
        return len(self.available_moves) == 0

    def make_child(self, network, move = None):
        if move is None: move = random.choice(self.available_moves)

        if move != -1:
            if self.to_play:
                new_black, new_white = o.update_board(move, self.black, self.white)
            else:
                new_white, new_black = o.update_board(move, self.white, self.black)
        else:
            # Pass -> boards unchanged
            new_white, new_black = self.white, self.black

        self.available_moves.remove(move)

        child = Node(network, self, new_white, new_black, self.turn_count + 1, move)
        self.children.append(child)

        return child

    def backpropogate(self, value):
        self.score += value
        self.visits += 1

        current_node = self.parent
        current_value = -value
        while current_node is not None:
            current_node.score += current_value
            current_node.visits += 1

            current_value *= -1
            current_node = current_node.parent

    def determine_PUCT(self, c):
        q = 0
        if self.visits > 0:
            q = self.score / self.visits
        p = c * self.parent.probabilities[self.move_to_reach] * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + p


if __name__ == '__main__':
    model = AlphaZeroNet()

    mc = NeuralMonteCarlo(model)

    mc.run_iterations(20)

    print('hi')