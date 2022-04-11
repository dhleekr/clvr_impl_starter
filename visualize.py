from models import Encoder, Decoder
from sprites_datagen.moving_sprites import MovingSpriteDataset
from general_utils import AttrDict
from sprites_datagen.rewards import VertPosReward, HorPosReward, AgentXReward, AgentYReward

import torch

import cv2
from general_utils import make_image_seq_strip
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

T = 30

spec = AttrDict(
    resolution=64,
    max_seq_len=T,
    max_speed=0.05,         # total image range [0, 1]
    obj_size=0.2,           # size of objects, full images is 1.0
    shapes_per_traj=1,      # number of shapes per trajectory
    rewards=[AgentYReward],
)

dataset = MovingSpriteDataset(spec)



encoder1 = Encoder().to(device)
encoder1.load_state_dict(torch.load('./model/fig2_X_encoder.pt', map_location=torch.device('cpu')))

decoder1 = Decoder(T).to(device)
decoder1.load_state_dict(torch.load('./model/fig2_X_decoder.pt', map_location=torch.device('cpu')))

# encoder2 = Encoder().to(device)
# encoder2.load_state_dict(torch.load('./model/Encoder_AgentX.pt'))

# decoder2 = Decoder(T).to(device)
# decoder2.load_state_dict(torch.load('./model/Decoder_AgentX.pt'))



images = dataset[1]['images']
print(images.shape)
img =  make_image_seq_strip([(images[None, :].astype(np.float32)) * 255], sep_val=255.0)
cv2.imwrite("Ground_Truth.png", img[0].transpose(1, 2, 0))



output = encoder1(torch.tensor(images))
output = decoder1(output).cpu().detach().numpy()
print(output.shape)
print(output[0])
img =  make_image_seq_strip([(output[None, :].astype(np.float32)) * 255], sep_val=255.0)
cv2.imwrite("AgentX_img.png", img[0].transpose(1, 2, 0))


# output = encoder2(torch.tensor(images))
# output = decoder2(output).cpu().detach().numpy()
# print(output.shape)
# print(output[0])
# img =  make_image_seq_strip([(output[None, :].astype(np.float32) + 1.0) * (255. / 2)], sep_val=255.0)
# cv2.imwrite("AgentX_img.png", img[0].transpose(1, 2, 0))
