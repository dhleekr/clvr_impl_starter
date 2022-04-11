from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict
from sprites_datagen.rewards import AgentXReward, AgentYReward, TargetXReward, TargetYReward

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import Encoder, MLP, LSTM, Network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

"""
Hyperparameter
"""
BATCH_SIZE = 1
RESOLUTION = 64
T = 40
REWARDS = [AgentXReward, AgentYReward, TargetXReward, TargetYReward]
LEARNING_RATE = 1e-3

"""
Dataset, Dataloader
"""
spec = AttrDict(
        resolution=RESOLUTION,
        max_seq_len=T,
        max_speed=0.05,
        obj_size=0.2,           
        shapes_per_traj=2,      
        rewards=REWARDS,
    )

dataset = MovingSpriteDataset(spec)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

""""
Model define
"""
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load('./model/reward_induced_encoder/encoder.pt', map_location=torch.device('cpu')))
mlp = MLP().to(device)
mlp.load_state_dict(torch.load('./model/reward_induced_encoder/mlp.pt', map_location=torch.device('cpu')))
lstm = LSTM().to(device)
lstm.load_state_dict(torch.load('./model/reward_induced_encoder/lstm.pt', map_location=torch.device('cpu')))
reward_heads = MLP(output_size=len(REWARDS)).to(device)
reward_heads.load_state_dict(torch.load('./model/reward_induced_encoder/reward_heads.pt', map_location=torch.device('cpu')))
model = Network(encoder, mlp, lstm, reward_heads, T)

res = []
for idx, sample in enumerate(dataloader):
    inputs = sample['images'][0]
    target_rewards = sample['rewards']
    target_r = []
    outputs = model(inputs)
    print(outputs.shape)

    for i, key in enumerate(target_rewards.keys()):
        target_r.append(target_rewards[key])
        print(nn.MSELoss()(target_rewards[key], outputs[:, :, i]))
