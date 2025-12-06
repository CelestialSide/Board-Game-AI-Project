import torch
import Othello
import Data
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Implements the value network component of our MCTS variant.
# This network's goal is to evaluate the win condition (value) of the current board state.
class ValueNetwork(nn.Module):

    def __init__(self):
        super(ValueNetwork, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(2, 8, 3), # 2 x 8 x 8  ->  8 x 6 x 6
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3), # 8 x 6 x 6  ->  16 x 4 x 4
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3), # 16 x 4 x 4  ->  32 x 2 x 2
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            # Flatten -> 128
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Tanh()
        )

    def forward(self, x):
        batch_size, _, _, _ = x.size()
        x = self.convolutional(x).view(batch_size, 128)
        return self.mlp(x)

def score_network(net, dat):
    num_correct = 0

    with torch.no_grad():
        dat_loader = DataLoader(dat, batch_size=32)
        p_bar = tqdm(dat_loader, desc="Scoring Network...")

        net = net.eval()
        for _, batch in enumerate(p_bar):
            labels, boards = batch

            pred = net(boards)
            # logits = torch.softmax(logits, dim=0)
            # pred = torch.argmax(logits, dim=1)

            for i in range(len(labels)):
                val = 0
                if pred[i, 0] > 0.1:
                    val = 1
                elif pred[i, 0] < -0.1:
                    val = -1

                if val == int(labels[i].item()):
                    num_correct += 1

            # num_correct += torch.sum(labels == pred)

    accuracy = (num_correct / len(dat))
    return accuracy

if __name__ == "__main__":
    epochs = 15
    batch_size = 64

    dat = Data.OthelloGames(path="cleaned_games.csv", train=True)
    net = ValueNetwork()

    optimizer = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        dat_loader = DataLoader(dat, batch_size, False)
        p_bar = tqdm(dat_loader, desc=f"Epoch {epoch+1}")

        for _, batch in enumerate(p_bar):
            labels, boards = batch

            optimizer.zero_grad()

            logits = net(boards).view(len(labels))

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            p_bar.set_postfix({'loss:' : loss.item()})

        torch.save(net.state_dict(), 'value_net.pt')
        print(score_network(net, Data.OthelloGames(path="cleaned_games.csv", train=False, run_full_game=False)))

    # dat = Data.OthelloGames(path="cleaned_games.csv", train=False, run_full_game=False)
    # net = ValueNetwork()
    # net.load_state_dict(torch.load('value_net.pt'))
    #
    # print(score_network(net, dat))

