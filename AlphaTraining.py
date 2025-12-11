import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from TrainingDataGeneration import PlayDataset
from tqdm import tqdm
from AlphaZeroNetwork import AlphaZeroNet

def train(network : AlphaZeroNet, play_dat : PlayDataset, batch_size, epochs, epochs_per_play=3, lr=1e-3, mcts_steps_per_turn=100,
          games_per_epoch=100, net_save_path="Models/zero.pt", dat_save_path="Data/zero_games.json",
          pre_train_policy_only = False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = network.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # For freezing value head
    if pre_train_policy_only:
        for param in net.conv_value.parameters():
            param.requires_grad = False
        for param in net.bn_value.parameters():
            param.requires_grad = False
        for param in net.fc_value1.parameters():
            param.requires_grad = False
        for param in net.fc_value2.parameters():
            param.requires_grad = False

    value_loss_fn = nn.MSELoss()

    dat = play_dat

    for epoch in range(epochs):
        # First, train based off of the current dataset.
        dat_loader = DataLoader(dat, batch_size, shuffle=True)
        p_bar = tqdm(dat_loader, desc=f"Epoch {epoch+1}")
        net = net.train()

        for _, batch in enumerate(p_bar):
            state, policy, value = batch

            optimizer.zero_grad()

            pred_p, pred_v = net(state.to(device))

            # Compute policy loss -> comparing policy distributions (slightly different that KL-Div)
            log_probs = torch.log_softmax(pred_p, dim=1)
            policy_loss = -(policy * log_probs).sum(dim=1).mean()

            # Compute value loss -> how good is the current state?
            value_loss = value_loss_fn(pred_v, value)

            # Combined loss is both policy and value loss.
            loss = policy_loss + value_loss

            # Optimize!
            loss.backward()
            optimizer.step()

            p_bar.set_postfix({'loss:' : loss.item(), 'policy_loss:' : policy_loss.item(), 'value_loss' : value_loss.item()})

        # Now, update the dataset by playing games. Save the new dataset and network after for later use.
        if not pre_train_policy_only:
            dat.play_games(net.eval(), games_per_epoch, mcts_steps_per_turn)
            dat.save_as(dat_save_path)
        torch.save(net.state_dict(), net_save_path)

    # For unfreezing value head
    if pre_train_policy_only:
        for param in net.conv_value.parameters():
            param.requires_grad = True
        for param in net.bn_value.parameters():
            param.requires_grad = True
        for param in net.fc_value1.parameters():
            param.requires_grad = True
        for param in net.fc_value2.parameters():
            param.requires_grad = True

    return net


if __name__ == '__main__':
    net = AlphaZeroNet()
    # net.load_state_dict(torch.load('Models/zero.pt'))
    dat = PlayDataset('Data/expert_start.json', max_buffer_size=130000)

    train(net, dat, 128, 100, 3, 3e-4, games_per_epoch=100)