import Othello
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import Play as p
import random
from tqdm import tqdm

class OthelloGames(Dataset):

    def __init__(self, path="othello_dataset.csv", run_full_game=False, train=True):
        self.csv = pd.read_csv(path)

        if train:
            self.csv = self.csv.iloc[:int(0.8*len(self.csv))]
        else:
            self.csv = self.csv.iloc[int(0.8*len(self.csv)):]

        self.run_full_game = run_full_game

    def __getitem__(self, item, run_full=False):
        # Win Condition Meanings:
        # -1 - White wins
        # 0 - Draw
        # 1 - Black wins
        win_condition = self.csv.iloc[item, 1] + 1

        moves = self.csv.iloc[item, 2]
        moves = [p.convert_move_to_index(moves[2*i:2*i+2]) for i in range(len(moves)//2)]

        # Generate a random length to go to in the game.
        if not self.run_full_game:
            rand_len = random.randint(0, len(moves))
        else:
            rand_len = len(moves)

        # Play out to this point in the game to get white & black bitboards
        white, black = 68853694464, 34628173824

        turn_count = 0
        move_dex = 0

        while move_dex < rand_len:
            if turn_count > 200:
                # Invalid game unique output
                return torch.tensor(10, dtype=torch.long), torch.zeros(1)

            if turn_count % 2 == 0:
                player, opponent = black, white
            else:
                player, opponent = white, black

            possible_moves = Othello.get_valid_move_list(player, opponent)

            # If move not valid, pass
            if moves[move_dex] not in possible_moves:
                turn_count += 1

            # Else, update board states
            else:
                player, opponent = Othello.update_board(moves[move_dex], player, opponent)

                if turn_count % 2 == 0:
                    black, white = player, opponent
                else:
                    white, black = player, opponent

                move_dex += 1
                turn_count += 1

        # Now, we need to convert those bitboards from numbers to structured data for a CNN
        # This is a 2 x 8 x 8 tensor, which is essentially the two game boards stacked on top
        # of each other.
        board = torch.zeros((2,8,8))

        for i in range(8):
            for j in range(8):
                if Othello.read_bit(black, i*8+j):
                    board[0, i, j] = 1
                elif Othello.read_bit(white, i*8+j):
                    board[1, i, j] = 1

        return torch.tensor(win_condition, dtype=torch.float32), board

    def __len__(self):
        return len(self.csv)



if __name__ == "__main__":
    # dat = OthelloGames(path="cleaned_games.csv")
    dat = OthelloGames(run_full_game=True)
    # print(len(dat))

    dat_loader = DataLoader(dat, batch_size=1)

    p_bar = tqdm(dat_loader, desc="Checking for erroneous moves.")
    count = 0
    indices_to_remove = []
    for i, batch in enumerate(p_bar):
        a = batch[0]

        if a[0].item() == 10:
            indices_to_remove.append(i)
            count += 1

    clean_df = dat.csv.drop(index=indices_to_remove)
    clean_df.to_csv("cleaned_games.csv", index=False)

    print(count)

