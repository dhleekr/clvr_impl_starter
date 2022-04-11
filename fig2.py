from models import Encoder, Decoder
from pyexpat import model
from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict
from sprites_datagen.rewards import VertPosReward, HorPosReward, AgentXReward, AgentYReward

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

"""
Hyperparameter
"""
BATCH_SIZE = 1
RESOLUTION = 64
T = 50
REWARDS = [AgentYReward]
LEARNING_RATE = 1e-3
EPOCH = 5

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

""""
Model define
"""
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load('./model/reward__prediction_encoder.pt'))

decoder = Decoder(T).to(device)

criterion = nn.MSELoss()
optimizer = optim.RAdam(decoder.parameters(), lr=LEARNING_RATE)

min_epoch_loss = 1e10
for epoch in range(EPOCH):
    decoder.train()
    
    epoch_loss = 0
    for idx, sample in enumerate(dataloader):
        inputs = sample['images'][0]
        # gradient를 0으로 초기화
        optimizer.zero_grad() 

        outputs = encoder(inputs)
        outputs = decoder(outputs)

        loss = criterion(outputs * 10, inputs * 10)
        epoch_loss += loss
        loss.backward()  

        optimizer.step()

        if idx % 100 == 0:
            print(f"{idx + 1} step loss : {loss.item()}")
    print(f"{epoch + 1} eopch loss : {epoch_loss.item() / len(dataloader)}")

    if min_epoch_loss > epoch_loss / len(dataloader):
        min_epoch_loss = epoch_loss / len(dataloader)
        torch.save(decoder.state_dict(), f'./model/reward_prediction_decoder.pt')
        print('New model saved!')