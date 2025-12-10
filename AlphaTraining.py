import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from TrainingDataGeneration import PlayDataset
from tqdm import tqdm
from AlphaZeroNetwork import AlphaZeroNet

def train(network, play_dat : PlayDataset, batch_size, epochs, lr=1e-3, mcts_steps_per_turn=100,
          games_per_epoch=100, net_save_path="Models/zero.pt", dat_save_path="Data/zero_games.json"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = network.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

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
        dat.play_games(net.eval(), games_per_epoch, mcts_steps_per_turn)
        dat.save_as(dat_save_path)
        torch.save(net.state_dict(), net_save_path)


if __name__ == '__main__':
    net = AlphaZeroNet()
    net.load_state_dict(torch.load('Models/zero.pt'))
    dat = PlayDataset('Data/zero_games.json', max_buffer_size=100000)
    # dat.play_games(net, num_games=600, mcts_its_per_turn=100)
    # dat.save_as('self_play.json')

    train(net, dat, 32, 100, 1e-3, games_per_epoch=60)