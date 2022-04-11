from models import Encoder, Decoder
from pyexpat import model
from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict
from sprites_datagen.rewards import VertPosReward, HorPosReward, AgentXReward, AgentYReward

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

"""
Hyperparameter
"""
BATCH_SIZE = 1
RESOLUTION = 64
T = 40
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
decoder = Decoder(T).to(device)

criterion = nn.MSELoss()
parameters = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.RAdam(parameters, lr=LEARNING_RATE)

writer = SummaryWriter('logs/image_reconstruction/')

min_epoch_loss = 1e10
t = 0
for epoch in range(EPOCH):
    encoder.train()
    decoder.train()
    
    epoch_loss = 0
    for idx, sample in enumerate(dataloader):
        t += 1
        inputs = sample['images'][0]

        optimizer.zero_grad() 

        outputs = encoder(inputs)
        outputs = decoder(outputs)

        loss = criterion(outputs, inputs)
        epoch_loss += loss
        loss.backward()  

        optimizer.step()
        
        writer.add_scalar("loss/train", loss.item(), t)

        if idx % 100 == 0:
            print(f"{idx + 1} step loss : {loss.item()}")
    print(f"{epoch + 1} eopch loss : {epoch_loss.item() / len(dataloader)}")

    if min_epoch_loss > epoch_loss / len(dataloader):
        min_epoch_loss = epoch_loss / len(dataloader)
        torch.save(encoder.state_dict(), f'./model/image_reconstruction_encoder.pt')
        print('New model saved!')

writer.flush()
writer.close()