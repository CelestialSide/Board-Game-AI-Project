import numpy as np
import random
import Othello

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

    def game(num_players):
        player, opponent = new_game()

        chosen_move = 0
        last_turn_pass = False
        turn = 0
        while True:
            match num_players:
                case 0:
                    chosen_move = get_move(player, opponent, True)
                case 1:
                    if turn % 2 == 0:
                        Othello.disp_game(opponent, player)
                        chosen_move = get_move(player, opponent, False)
                    else:
                        chosen_move = get_move(player, opponent, True)
                        print(f'CPU Chooses: {chr(chosen_move % 8 + 65)}{chosen_move // 8 + 1}')
                case 2:
                    if turn % 2 == 0:
                        Othello.disp_game(opponent, player)
                    else:
                        Othello.disp_game(player, opponent)
                    chosen_move = get_move(player, opponent, False)

            if chosen_move == -1:
                if last_turn_pass:
                    # Game complete! Parsing who is black currently
                    if turn % 2 == 0:
                        return opponent, player
                    else:
                        return player, opponent
                else:
                    last_turn_pass = True
            else:
                player, opponent = Othello.update_board(chosen_move, player, opponent)
                last_turn_pass = False

            player, opponent = opponent, player
            turn += 1


if __name__ == "__main__":
    print("Hello World")

