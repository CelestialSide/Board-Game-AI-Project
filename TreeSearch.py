import random
from copy import copy
import Board as b
import Othello as o
import numpy as np
from tqdm import tqdm

def random_game(white = 34628173824, black = 68853694464, turn_count = 0):
    if turn_count % 2 == 0: player, opponent = black, white
    else: player, opponent = white, black

    last_turn_pass = False
    while True:
        valid_moves = o.get_valid_move_list(player, opponent)
        if len(valid_moves) == 0: chosen_move = -1
        else: chosen_move = random.choice(valid_moves)

        if chosen_move == -1:
            if last_turn_pass:
                # Game complete! Parsing who is black currently
                #   Returns: White Board, Black Board
                if turn_count % 2 == 0:
                    return o.determine_winner(opponent, player)[1]
                else:
                    return o.determine_winner(player, opponent)[1]
            else:
                last_turn_pass = True
        else:
            player, opponent = o.update_board(chosen_move, player, opponent)
            last_turn_pass = False

        player, opponent = opponent, player
        turn_count += 1


class MonteCarlo:
    def __init__(self, root):
        self.root = root
    def get_best_child(self, node):
        while node.is_explored():
            utc = [(child, child.compute_UCT(c = 2)) for child in node.children]
            max_utc = max([child[1] for child in utc])
            selected_node = random.choice([child for child in utc if child[1] == max_utc])[0]
            node = selected_node
        return node

    def expand(self, parent):
        child = parent.make_child()
        return child

    def rollout(self, node, reward=1):
        """Play random moves until the game ends."""
        result = random_game(node.white, node.black, node.turn_count)
        if result > 0:
            return reward
        elif result < 0:
            return -1 * reward
        else:
            return 0

    def backpropagation(self, node, score):
        while node.parent is not None:
            if node.to_play:
                node.score += score
            else:
                node.score += -score
            node.visits += 1
            node = node.parent

def monte_carlo_tree_search(root=None, iterations=1000):
    if root is None:
        true_root = b.Node()
        true_root.visits = 1
        tree = MonteCarlo(true_root)
    else:
        if root.visits < 1: root.visits = 1
        tree = MonteCarlo(root)

    progress_bar = tqdm(range(iterations))
    for iteration in progress_bar:
        node = tree.get_best_child(tree.root)
        child = tree.expand(node)
        score = tree.rollout(child)
        tree.backpropagation(child, score)

        if (iteration + 1) % 100 == 0:
            progress_bar.set_postfix({'Top Move': tree.root.compute_best_score()[1]})
    return tree.root.compute_best_score()

if __name__ == "__main__":
    # Test Code running through 2 MCTS moves
    white, black = 34628173824, 68853694464
    node, move = monte_carlo_tree_search()
    black, white = o.update_board(move, black, white)
    #o.disp_game(white, black, False)

    valid_moves = o.get_valid_move_list(white, black)
    move = random.choice(valid_moves)
    white, black = o.update_board(move, white, black)
    #o.disp_game(white, black, True)

    if move in node.available_moves:
        new_root = node.make_child(move)
    else:
        for child in node.children:
            if child.move == move:
                new_root = child
                break

    node, move = monte_carlo_tree_search(new_root)
    black, white = o.update_board(move, black, white)
