import torch
import torch.nn as nn
import numpy as np
from models import Encoder
from distribution import DiagGaussianDistribution


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, observation_space, out_dim, mode=''):
        super(CNN, self).__init__()
        self.out_dim = out_dim
        n_input_channels = 1
        self.w, self.h = observation_space.shape[0], observation_space.shape[1]
        self.out_dim = out_dim
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            cnn_size = self.cnn(torch.as_tensor(observation_space.sample()[None][None]).float()).shape
            n_flatten = cnn_size[0] * cnn_size[1]
        self.features_dim = n_flatten

        self.policy_net = nn.Sequential(
            nn.Linear(n_flatten, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim)
        )

        self.value_net = nn.Sequential(
            nn.Linear(n_flatten, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(device)
        obs = obs.view(-1, 1, self.w, self.h)
        shared_latent = self.cnn(obs)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


class Representation(nn.Module):
    def __init__(self, observation_space, out_dim, mode='Image_scratch'):
        super(Representation, self).__init__()
        self.w, self.h = observation_space.shape[0], observation_space.shape[1]
        self.out_dim = out_dim
        self.encoder = Encoder().to(device)

        if mode == 'Image_scratch':
            pass
        elif mode[0] == 'I':
            self.encoder.load_state_dict(torch.load(f'./model/Image_reconstruction_encoder.pt', map_location=torch.device('cpu')))
        else:
            self.encoder.load_state_dict(torch.load(f'./model/Reward_prediction_encoder.pt', map_location=torch.device('cpu')))

        self.policy_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(device)
        obs = obs.view(-1, 1, self.w, self.h)
        shared_latent = self.encoder(obs)
        return self.policy_net(shared_latent), self.value_net(shared_latent)


class Oracle(nn.Module):
    def __init__(self, observation_space, out_dim, mode=''):
        super(Oracle, self).__init__()
        self.out_dim = out_dim
        in_dim = observation_space.shape[0]
        self.policy_net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim)
        )

        self.value_net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(device)
        return self.policy_net(obs), self.value_net(obs)


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, mode=''):
        super(ActorCritic, self).__init__()
        if mode == 'CNN':
            self.features_extractor = CNN(observation_space, 64)
        elif mode == 'Oracle':
            self.features_extractor = Oracle(observation_space, 64)
        else:
            self.features_extractor = Representation(observation_space, 64, mode=mode)

        self.action_dist = DiagGaussianDistribution(action_space.shape[0])
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(latent_dim=self.features_extractor.out_dim, log_std_init=0.0)


    def forward(self, obs):
        latent_pi, values = self.features_extractor(obs)
        mean_actions = self.action_net(latent_pi)
        dist = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = dist.sample()
        log_prob = dist.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        latent_pi, values = self.features_extractor(obs)
        mean_actions = self.action_net(latent_pi)
        dist = self.action_dist.proba_distribution(mean_actions, self.log_std)
        actions = actions.view(-1, 2)
        log_prob = dist.log_prob(actions).view(-1, 1)
        return values, log_prob, dist.entropy()

