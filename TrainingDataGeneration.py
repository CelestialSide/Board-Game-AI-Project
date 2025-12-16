import Othello as o
import random
import math
import torch
from AlphaZeroNetwork import AlphaZeroNet
from NeuralMonteCarlo import board_state_to_tensor, NeuralMonteCarlo
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import numpy as np

"""
Goal: Generate a bunch of training data from an AlphaZero-style network trying to learn to
play Othello.
"""

def generate_game_data(network, mcts_its_per_turn=100, mcts_batch_size=8):
    network = network.eval()
    with torch.no_grad():
        # Initalize board state
        white = 68853694464
        black = 34628173824
        turn_count = 0
        pass_last_turn = False

        data_white = []
        data_black = []
        data_turn = []
        data_policy = []

        mcts = NeuralMonteCarlo(network, white, black, turn_count, train_mode=True)

        # Loop through the game's logic until it is over.
        while True:
            turn = turn_count % 2 == 0

            # Evaluate MCTS for the current move -> Generates a policy probability distribution.
            # This distribution is stored as a dictionary.
            # Run mcts_its_per_turn simulations on the tree.
            mcts.run_iterations(mcts_its_per_turn, mcts_batch_size)

            # Add policy distribution and game state into data.
            dist = mcts.get_root_visit_distribution()

            data_white.append(white)
            data_black.append(black)
            data_turn.append(turn_count)
            data_policy.append(dist)

            # For early on in the game, play creatively
            temp_factor = 0
            if turn_count < 10:
                temp_factor = 1.0
            elif turn_count < 20:
                temp_factor = 0.5
            elif turn_count < 30:
                temp_factor = 0.25
            elif turn_count < 40:
                temp_factor = 0.1

            move = mcts.get_move_to_play(temperature=temp_factor)

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

            # Update the tree after a move has been made
            mcts.shift_root(white, black, turn_count, move)

        # Now that the game is over, determine winner and add the winner in each data entry.
        value = o.determine_winner(black, white) # 1 if black won, -1 if white won

        data_value = [0] * len(data_turn)
        for i in range(len(data_turn)):
            turn = data_turn[i]

            # Decide whether the current player won
            v = value
            if not turn % 2 == 0:
                v *= -1

            data_value[i] = v
        
        data_white = np.array(data_white, dtype=np.uint64)
        data_black = np.array(data_black, dtype=np.uint64)
        data_turn = np.array(data_turn, dtype=np.uint16)
        data_policy = np.array(data_policy, dtype=np.float32)
        data_value = np.array(data_value, dtype=np.float32)

        return [data_white, data_black, data_turn, data_policy, data_value]

def add_to_buffer(buffer, games, max_buffer_size=100000):
    for i in range(len(buffer)):
        if i != 3:
            buffer[i] = np.append(buffer[i], games[i])
        else:
            buffer[i] = np.concatenate([buffer[i], games[i]], axis=0)

    if len(buffer[i]) > max_buffer_size:
        dif = len(buffer[i]) - max_buffer_size

        for i in range(len(buffer)):
            buffer[i] = buffer[i][dif:]

    return buffer

def add_games_to_buffer(buffer, network, num_games, max_buffer_size=100000, mcts_its_per_turn=100, mcts_batch_size=8):
    games = None
    p_bar = tqdm(range(num_games), desc="Playing out Games")

    for i in p_bar:
        dat = generate_game_data(network, mcts_its_per_turn, mcts_batch_size)
        
        if games is None:
            games = dat
        else:
            for k in range(len(buffer)):
                if k != 3:
                    games[k] = np.append(games[k], dat[k])
                else:
                    games[k] = np.concatenate([games[k], dat[k]], axis=0)

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
            self.read(filepath, cap=pre_load_cap)

        if self.buffer is None:
            self.buffer = [np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint64), np.empty(0, dtype=np.uint16), np.empty((0,65), dtype=np.float32), np.empty(0, dtype=np.float32)]

    def __len__(self):
        return len(self.buffer[0])

    def __getitem__(self, item):
        # white, black, turn_count, dist, value = self.buffer[item]
        white = self.buffer[0][item]
        black = self.buffer[1][item]
        turn_count = self.buffer[2][item]
        dist = self.buffer[3][item]
        value = self.buffer[4][item]

        # Load board state as a tensor
        state = board_state_to_tensor(int(white), int(black), int(turn_count))[0]

        # Translate the policy distribution into a tensor to get the policy target
        policy = torch.tensor(dist, dtype=torch.float32)

        # for key in dist.keys():
        #     if key != -1:
        #         policy[int(key)] = dist[key]
        #     else:
        #         policy[64] = dist[key]

        # Convert the value into a torch datatype
        value = torch.tensor(value, dtype=torch.float32)

        return state, policy, value

    def play_games(self, network, num_games=60, mcts_its_per_turn=100, mcts_batch_size=8):
        self.buffer = add_games_to_buffer(self.buffer, network, num_games, self.max_buffer_size, mcts_its_per_turn, mcts_batch_size)

    def save_as(self, filepath):
        # with open(filepath, 'w', encoding='utf-8') as file:
        #     json.dump(self.buffer, file)

        np.savez(filepath, white=self.buffer[0], black=self.buffer[1], turn=self.buffer[2], policy=self.buffer[3], value=self.buffer[4])

    def read(self, filepath, cap):
        # with open(filepath, 'r', encoding='utf-8') as file:
        #     self.buffer = json.load(file)
        
        data = np.load(filepath)

        self.buffer = []
        self.buffer.append(data['white'])
        self.buffer.append(data['black'])
        self.buffer.append(data['turn'])
        self.buffer.append(data['policy'])
        self.buffer.append(data['value'])

        if cap != -1 and len(self.buffer[0]) > cap:
            for i in range(len(self.buffer)):
                dif = len(self.buffer[i]) - cap
                self.buffer[i] = self.buffer[i][dif:]



if __name__ == '__main__':
    model = AlphaZeroNet()

    # dat = PlayDataset(filepath="Data/expert_start.npz")
    dat = PlayDataset()
    dat.play_games(model, 5, mcts_its_per_turn=2)

    # dat.save_as("Data/self_play.npz")

    for i in range(len(dat)):
        hi = dat[i]
        print('hi')