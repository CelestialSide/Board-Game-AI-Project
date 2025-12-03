import numpy as np
import random
import Othello as o
import Board as b


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
                    return o.determine_winner(opponent, player)
                else:
                    return o.determine_winner(player, opponent)
            else:
                last_turn_pass = True
        else:
            player, opponent = o.update_board(chosen_move, player, opponent)
            last_turn_pass = False

        player, opponent = opponent, player
        turn_count += 1

# Node __init__(self, parent=None, white=34628173824, black=68853694464, turn_count=0)
class MonteCarlo:
    def __init__(self, actions):
        self.root = b.Node()

    def selection(self, node):
        if not node.is_explored():
            return node
        else:
            utc = [(child, child.compute_UCT) for child in node.children]
            max_utc = max([child[1] for child in utc])
            selected_node = random.choice([child for child in utc if child[1] == max_utc])[0]
            final_node = self.selection(selected_node)
        return final_node


    def expansion(self, parent):
        child = parent.make_child(parent)
        return child


    def simulation(self, node):
        score = random_game(node.white, node.black, node.turn_count)
        return score


    def backpropagation(self, node, score):
        if node.parent is not None:
            if node.to_play:
                node.score += -score
            else:
                node.score += score
            node.visits += 1

            self.backpropagation(node.parent, node.score)






