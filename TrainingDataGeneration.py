import Othello as o
import random
import math
import torch
from AlphaZeroNetwork import AlphaZeroNet
from NeuralMonteCarlo import board_state_to_tensor, NeuralMonteCarlo

"""
Goal: Generate a bunch of training data from an AlphaZero-style network trying to learn to
play Othello.
"""

def generate_game_data(network, mcts_its_per_turn=100):
    network = network.eval()
    with torch.no_grad():
        # Initalize board state
        white = 34628173824
        black = 68853694464
        turn_count = 0
        pass_last_turn = False

        data = []

        # Loop through the game's logic until it is over.
        while True:
            turn = turn_count % 2 == 0

            # Evaluate a MCTS for the current move -> Generates a policy probability distribution.
            # This distribution is stored as a dictionary.
            mcts = NeuralMonteCarlo(network, white, black, turn_count)

            # Run mcts_its_per_turn simulations on the tree.
            mcts.run_iterations(mcts_its_per_turn)

            # Add distribution and game state into data.
            dist = mcts.get_root_visit_distribution()
            data.append([white, black, turn_count, dist])

            move = mcts.get_move_to_play()

            # See if we're at the end
            if move == -1:
                if pass_last_turn:
                    break # End Game
                else:
                    pass_last_turn = True
            else:
                pass_last_turn = False

                # If we don't pass, play that move
                if turn:
                    black, white = o.update_board(move, black, white)
                else:
                    white, black = o.update_board(move, white, black)

            turn_count += 1

        # Now that the game is over, determine winner and add the winner in each data entry.
        value = o.determine_winner(black, white) # 1 if black won, -1 if white won

        for i in range(len(data)):
            turn = data[i][2]

            # Decide whether the current player won
            v = value
            if not turn % 2 == 0:
                v *= -1

            data[i].append(v)

        return data

if __name__ == '__main__':
    model = AlphaZeroNet()

    dat = generate_game_data(model, mcts_its_per_turn=100)

    print('hi')