import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(out + x)

class AlphaZeroNet(nn.Module):
    def __init__(self, in_channels = 3, feature_maps = 2, num_res_blocks = 6, channels = 64):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)
        # res blocks
        res_blocks_list = []
        for _ in range(num_res_blocks):
            res_blocks_list.append(ResBlock(channels))

        self.res_blocks = nn.Sequential(*res_blocks_list)
        # policy head
        self.conv_policy = nn.Conv2d(channels, feature_maps, kernel_size = 1)
        self.bn_policy = nn.BatchNorm2d(feature_maps)
        self.fc_policy = nn.Linear(feature_maps * 64, 65) # feature_maps * board_dim * board_dim --> 65 logits
        # value head
        self.conv_value = nn.Conv2d(channels, 1, kernel_size = 1)
        self.bn_value = nn.BatchNorm2d(1)
        self.fc_value1 = nn.Linear(64, 256) #board_dim * board_dim
        self.fc_value2 = nn.Linear(256, 1)

    def forward(self, x):
        # Res block
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        # policy
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p) # (B, 65)

        #value
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value1(v))
        v = torch.tanh(self.fc_value2(v).squeeze(-1))

        return p,v

def train(batch_size = 32, num_epochs = 10):
    x = torch.randn(100, 3, 8, 8)
    pi_target = torch.rand(100, 65)
    pi_target = pi_target / pi_target.sum(dim=1, keepdim=True)  # normalize
    v_target = torch.randn(100).clamp(-1, 1)
    # x, pi_target, v_target = training_set # input from MCTS

    dataset = TensorDataset(x, pi_target, v_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AlphaZeroNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #Loss functions
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    #Training loop
    for epoch in range(num_epochs):
        model.train()
        p_bar = tqdm(loader, desc=f"Epoch {epoch + 1}")
        for _, batch in enumerate(p_bar):
            states, pi, v = [b.to(device) for b in batch]
            pred_pi, pred_v = model(states)
            pi_target = pi.argmax(dim=1)
            loss_policy = policy_loss_fn(pred_pi, pi_target)
            loss_value = value_loss_fn(pred_v, v)
            loss = loss_policy + loss_value
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            p_bar.set_postfix({'policy loss': loss_policy.item(), 'value loss': loss_value.item()})

if __name__ == '__main__':
    train()