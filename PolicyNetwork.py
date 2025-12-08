import torch
import Data_Policy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Policy network takes in board state
# evaluate probability of win for possible next moves

class PolicyNetwork(nn.Module):
    def __init__(self, in_channels = 2):
        super().__init__()
        self.conv =  nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = self.conv(x)
        logits = self.policy_head(x)
        return F.softmax(logits, dim=1)


def train_policy_net(policy_net, dataset, epochs=10, batch_size=32):
    loader = DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.Adam(policy_net.parameters(), 1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        p_bar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for _, batch in enumerate(p_bar):
            target_policy, boards = batch
            optimizer.zero_grad()

            pred_policy = policy_net(boards)
            loss = loss_fn(target_policy, pred_policy)

            loss.backward()
            optimizer.step()

            p_bar.set_postfix({'loss': loss.item()})

        torch.save(policy_net.state_dict(), "policy_net.pt")


if __name__ == "__main__":
    data = Data_Policy.OthelloGames(path="cleaned_games.csv", train=True)
    train_policy_net(PolicyNetwork(), data)


