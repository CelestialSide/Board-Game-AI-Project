import Othello as o
import random
import math
import torch
from AlphaZeroNetwork import AlphaZeroNet
from NeuralMonteCarlo import board_state_to_tensor, NeuralMonteCarlo
from tqdm import tqdm
import json
from torch.utils.data import Dataset

"""
Goal: Generate a bunch of training data from an AlphaZero-style network trying to learn to
play Othello.
"""

def generate_game_data(network, mcts_its_per_turn=100):
    network = network.eval()
    with torch.no_grad():
        # Initalize board state
        white = 68853694464
        black = 34628173824
        turn_count = 0
        pass_last_turn = False

        data = []

        # Loop through the game's logic until it is over.
        while True:
            turn = turn_count % 2 == 0

            # Evaluate MCTS for the current move -> Generates a policy probability distribution.
            # This distribution is stored as a dictionary.
            mcts = NeuralMonteCarlo(network, white, black, turn_count)

            # Run mcts_its_per_turn simulations on the tree.
            mcts.run_iterations(mcts_its_per_turn)

            # Add policy distribution and game state into data.
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

def add_to_buffer(buffer, games, max_buffer_size=100000):
    buffer = buffer + games

    if len(buffer) > max_buffer_size:
        dif = len(buffer) - max_buffer_size

        buffer = buffer[dif:]

    return buffer

def add_games_to_buffer(buffer, network, num_games, max_buffer_size=100000, mcts_its_per_turn=100):
    games = []
    p_bar = tqdm(range(num_games), desc="Playing out Games")

    for i in p_bar:
        dat = generate_game_data(network, mcts_its_per_turn=mcts_its_per_turn)

        games = games + dat

    return add_to_buffer(buffer, games, max_buffer_size=max_buffer_size)

class PlayDataset(Dataset):
    """
    This class houses a dataset that contains a replay buffer of games an AlphaZero model has played.
    By training off of these games, it learns to improve itself.
    """

    def __init__(self, filepath=' ', buffer=None, max_buffer_size=60000, pre_load_cap=40000):
        self.buffer = buffer
        self.max_buffer_size = max_buffer_size

        if filepath != ' ' and self.buffer is None:
            self.read(filepath, pre_load_cap)

        if self.buffer is None:
            self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, item):
        white, black, turn_count, dist, value = self.buffer[item]

        # Load board state as a tensor
        state = board_state_to_tensor(white, black, turn_count)[0]

        # Translate the policy distribution into a tensor to get the policy target
        policy = torch.zeros((65,), dtype=torch.float32)

        for key in dist.keys():
            if key != -1:
                policy[int(key)] = dist[key]
            else:
                policy[64] = dist[key]

        # Convert the value into a torch datatype
        value = torch.tensor(value, dtype=torch.float32)

        return state, policy, value

    def play_games(self, network, num_games=60, mcts_its_per_turn=100):
        self.buffer = add_games_to_buffer(self.buffer, network, num_games, self.max_buffer_size, mcts_its_per_turn)

    def save_as(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(self.buffer, file)

    def read(self, filepath, cap=-1):
        with open(filepath, 'r', encoding='utf-8') as file:
            self.buffer = json.load(file)
        
        if cap != -1 and len(self.buffer) > cap:
            self.buffer = self.buffer[:cap]



if __name__ == '__main__':
    model = AlphaZeroNet()

    dat = PlayDataset(filepath="Data/self_play.json")
    dat.play_games(model, 40)

    dat.save_as("self_play.json")