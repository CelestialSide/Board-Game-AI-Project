import random
from copy import copy

import numpy as np

class TreeNode:
    def __init__(self, data, parent = None, action = None):
        self.state = data

        self.children = []
        self.parent = parent
        self.action = action
        self.untried_actions = open_spots(data)

        self.score = 0
        self.traversals = 0

    def add_traversal(self):
        self.traversals += 1

    def add_child(self, child_node):
        child_node.parent = self
        self.children.append(child_node)

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent
        return level

    def get_current_player(self):
        return 2 if self.state.count(1) > self.state.count(2) else 1

    def print_tree(self):
        prefix = "  " * self.get_level() + "|-- " if self.parent else ""
        print(prefix + str(self.state) + f" n: {self.traversals}, t: {self.score}")
        if self.children:
            for child in self.children:
                child.print_tree()

    def expand(self):
        """Add one of the remaining actions as a child."""
        picked_move = self.untried_actions.pop()
        new_state = copy(self.state)
        player = self.get_current_player()
        new_state[picked_move] = player
        child = TreeNode(new_state, parent=self, action=picked_move)
        self.children.append(child)
        return child

    def game_over(self, state = None):
        used_state = self.state if not state else state
        prev_spot = used_state[0]
        for i in range(1, len(used_state)):
            if not prev_spot == 0 and used_state[i] == prev_spot:
                return used_state[i]
            else:
                prev_spot = used_state[i]
        if len(self.untried_actions) == 0:
            return -1
        return 0

    def best_child(self, const = 2):
        if self.children:
            best_child = self.children[0]
            for c in self.children:
                BUCB = best_child.score / best_child.traversals + const * np.sqrt(
                    np.log(self.traversals) / best_child.traversals)
                CUCB = c.score / c.traversals + const * np.sqrt(np.log(self.traversals) / c.traversals)
                if CUCB > BUCB:
                    best_child = c

            return best_child
        return self

    def rollout(self):
        """Play random moves until the game ends."""
        state = copy(self.state)
        player = self.get_current_player()

        while True:
            winner = self.game_over(state = state)
            if not winner == 0 and not winner == -1:
                return -1 if winner == 1 else 1

            actions = open_spots(state)
            if not actions:
                return 0  # Draw

            rand_move = random.choice(actions)
            state[rand_move] = player
            player = 1 if player == 2 else 2

    def backpropagate(self, result):
        self.traversals += 1
        self.score += result
        if self.parent:
            self.parent.backpropagate(result)


def MCTS(board, iterations=100) -> int:
    root = TreeNode(board)

    for i in range(iterations):
        node = root
        current_game_state = node.game_over()

        # Will find the best child node as long as the game is still possible
        while current_game_state == 0 and len(node.untried_actions) == 0:
            node = node.best_child()

        # If the game has not ended, the best node is then explored.
        current_game_state = node.game_over()
        if current_game_state == 0:
            node = node.expand()

        result = node.rollout()

        node.backpropagate(result)

    return root.best_child(const=0).action

def score_board(board_state):
    if board_state == -1 or board_state == 0:
        return 0

    return -(board_state * 2 - 3)

def open_spots(board):
    open_spots = []
    for i in range(len(board)):
        if board[i] == 0:
            open_spots.append(i)

    return open_spots

def ask_for_move(board) -> int:
    print("Input move location(ex. 0 for the first spot): ")
    user_input = input()
    poss_moves = open_spots(board)
    while not validate_input(user_input, poss_moves):
        print("Input move location(ex. 0 for the first spot): ")
        user_input = input()

    return int(user_input)

def validate_input(i, pm):
    for m in pm:
        if i == str(m):
            return True
    return False

if __name__ == "__main__":
    game_board = [0,0,0,0]
    root = TreeNode(game_board)
    game_state = root.game_over()
    p = 1
    while game_state == 0:
        root = TreeNode(game_board)
        if p == 1:
            move = ask_for_move(game_board)
        else:
            move = MCTS(game_board)

        game_board[move] = p

        root.print_tree()
        print(game_board)

        p = 3 - p
        game_state = root.game_over()