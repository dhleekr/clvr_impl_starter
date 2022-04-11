from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict
from sprites_datagen.rewards import AgentXReward, AgentYReward, TargetXReward, TargetYReward, VertPosReward, HorPosReward

import torch
from torch.utils.data import DataLoader

from models import Encoder, MLP, LSTM, trainer

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
        max_speed=0.05,         # total image range [0, 1]
        obj_size=0.2,           # size of objects, full images is 1.0
        shapes_per_traj=2,      # number of shapes per trajectory
        rewards=REWARDS,
    )

dataset = MovingSpriteDataset(spec)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

""""
Model define
"""
encoder = Encoder().to(device)
mlp = MLP().to(device)
lstm = LSTM().to(device)
reward_heads = MLP(output_size=len(REWARDS)).to(device)

"""
Training Part
"""
model_trainer = trainer(encoder, mlp, lstm, reward_heads, dataloader, val_loader, epoch_size=5, learning_rate=LEARNING_RATE, max_len=T)
model_trainer.train()