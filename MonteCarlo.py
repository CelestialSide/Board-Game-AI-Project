import numpy as np
import random

class MonteCarlo:
    def __init__(self, actions):
        self.root = Node(np.zeros(4), None, None)

    def selection(self, node):
        if len(node.get_unvisited()) > 0:
            return random.choice(node.get_unvisited())
        else:
            utc = [(child, child.get_utc) for child in node.children]
            max_utc = max([child[1] for child in utc])
            selected_node = random.choice([child for child in utc if child[1] == max_utc])[0]
            final_node = self.selection(selected_node)
        return final_node


    def expansion(self, parent):
        next_move = random.choice(parent.get_unvisited())
        new_board = parent.board.copy()

        new_board[next_move] = 1
        new_node = Node(new_board, next_move, parent)

        parent.add_child(new_node)
        return new_node


    def simulation(self, node):
        simulated_board = node.board.copy()
        simulated_actions = get_available_actions(simulated_board)
        print(simulated_actions[0])

        while len(simulated_actions[0]) > 0:
            action = random.choice(simulated_actions[0])
            if np.sum(simulated_board) == 0:
                simulated_board[action] = 1
            else:
                simulated_board[action] = -1
            simulated_actions = get_available_actions(simulated_board)

            score = check_for_winner(simulated_board)
            if score != 0:
                return score
        return 0


    def backpropagation(self, node, state):
        if node.parent is not None:
            node.visits += 1
            node.state += state
            self.backpropagation(node.parent, node.state)

class Node:
    def __init__(self, board, move, parent):
        self.board = board
        self.move = move
        self.actions = get_available_actions(self.board)
        self.parent = parent

        self.visits = 0
        self.state = 0
        self.children = []

    def get_unvisited(self):
        child_moves = [child.move for child in self.children]
        return list(set(self.actions) - set(child_moves))

    def get_utc(self, C):
        if self.visits == 0: return np.inf
        Vi = self.state / self.visits
        return Vi + C * np.sqrt(2*np.log(self.parent.visits) / self.visits)

    def add_child(self, child):
        self.children.append(child)

def get_available_actions(board):
    return np.where(board == 0)

def check_for_winner(board):
    cpu_one_in_a_row = False
    mc_one_in_a_row = False

    for tile in board:
        if tile == -1:
            if cpu_one_in_a_row: return -1
            else:
                cpu_one_in_a_row = True
                mc_one_in_a_row = False
        if tile == 1:
            if mc_one_in_a_row: return 1
            else:
                mc_one_in_a_row = True
                cpu_one_in_a_row = False
        if tile == 0:
            cpu_one_in_a_row = False
            mc_one_in_a_row = False
    return 0

def game():
    board = np.zeros(4)
    player = 1

    for _ in range(4):
        actions = get_available_actions(board)

        if player == 1:
            i = 0
            # MONTE CARLO
        else:
            board[random.choice(actions)] = -1

        score = check_for_winner(board)
        if score != 0: return score
        player *= -1

    print("Draw")
    return 0

if __name__ == "__main__":
    print("Hello World")

