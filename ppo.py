import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from distribution import DiagGaussianDistribution

from ppo_networks import ActorCritic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, baseline_class, env, env_type, **hyperparameters):
        self._init_hyperparameters(hyperparameters)
        self.mode = baseline_class
        self.env_type = env_type

        self.writer = SummaryWriter(f'logs/ppo/{self.mode}/') # For recording

        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # Select Baseline
        self.actor_critic = ActorCritic(env.observation_space, env.action_space, mode=self.mode).to(device)

        # Select finetuning or not        
        if self.mode == 'Image_reconstruction' or self.mode == 'Reward_prediction':
            print('Layer Freezing...')
            for name, param in self.actor_critic.named_parameters():
                temp = name.split('.')
                if len(temp) > 1 and temp[1] == 'encoder':
                    param.requires_grad = False
        else:
            for param in self.actor_critic.parameters():
                param.requires_grad = True

        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.lr)
        self.max_grad_norm = 0.5

        self.logger = {
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],
            'entropy_losses' : [],
        }

    
    def learn(self, total_timesteps):
        print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
        print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
        self.t = 0
        t_so_far = 0
        i_so_far = 0
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            batch_obs, batch_acts, batch_log_probs, batch_rtgs = batch_obs.to(device), batch_acts.to(device), batch_log_probs.to(device), batch_rtgs.to(device)

            t_so_far += np.sum(batch_lens)
            i_so_far += 1
            self.logger['t_so_far'] = t_so_far
            self.logger['i_so_far'] = i_so_far

            with torch.no_grad():
                values, _, _ = self.actor_critic.evaluate_actions(batch_obs, batch_acts)
                values = values.view(-1)
            A_k = batch_rtgs - values
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
            
            for _ in range(self.n_updates_per_iteration):

                values, curr_log_probs, dist_entropy = self.actor_critic.evaluate_actions(batch_obs, batch_acts)
                values = values.view(-1)
                
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                ratios = ratios.view(-1)
                
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, batch_rtgs)
                entropy_loss = -torch.mean(dist_entropy)
           
                self.optimizer.zero_grad()

                total_loss = critic_loss * self.value_loss_coef + actor_loss + self.entropy_coef * entropy_loss

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

                self.optimizer.step()
                
                self.logger['actor_losses'].append(actor_loss.detach())
                self.logger['critic_losses'].append(critic_loss.detach())
                self.logger['entropy_losses'].append(entropy_loss.detach())

            self._log_summary()
            
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor_critic.state_dict(), f'./model/actor_critic_{self.mode}_{self.env_type}.pth')
        
        self.writer.flush()
        self.writer.close()



    def evaluate(self, batch_obs, batch_acts):
        action_features, values = self.actor_critic(batch_obs)

        dist = self.new_dist(action_features)

        log_probs = dist.log_probs(batch_acts)
        dist_entropy = dist.entropy().mean()

        return values.squeeze(), log_probs, dist_entropy

    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        t = 0
        while t < self.timesteps_per_batch:
            ep_rews = []
            obs = self.env.reset()
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                if self.render and (self.logger['i_so_far'] % self.render_every_i == 0) and len(batch_lens) == 0:
                    self.env.render()

                self.t += 1
                t += 1
                
                with torch.no_grad():
                    batch_obs.append(obs)
                    action, _, log_prob = self.actor_critic(obs)
                action, log_prob = action.cpu().numpy(), log_prob.cpu().numpy()

                obs, rew, done, _ = self.env.step(action)

                # Collect rewards, action, log probs
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob.reshape(-1))

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        with torch.no_grad():
            batch_rtgs = self.compute_rtgs(batch_rews)

        self.writer.add_scalar(f"{self.env_type}", np.mean([np.sum(ep_rews) for ep_rews in batch_rews]), self.t) # reward record

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        batch_rtgs = (batch_rtgs - batch_rtgs.mean()) / (batch_rtgs.std() + 1e-10)
        return batch_rtgs

    def _init_hyperparameters(self, hyperparameters):
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 1600
        self.n_updates_per_iteration = 10
        self.gamma = 0.95
        self.clip = 0.1
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.lr = 0.005
        self.save_freq = 10
        self.render = False
        self.render_every_i = 10

        for param, val in hyperparameters.items():
            exec('self.' + param + ' = ' + str(val))

    def _log_summary(self):
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.to('cpu').float().mean() for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.to('cpu').float().mean() for losses in self.logger['critic_losses']])
        avg_entropy_loss = np.mean([losses.to('cpu').float().mean() for losses in self.logger['entropy_losses']])

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_critic_loss = str(round(avg_critic_loss, 5))
        avg_entropy_loss = str(round(avg_entropy_loss, 5))
        

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
        print(f"Average Entropy Loss: {avg_entropy_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        self.logger['entropy_losses'] = []





        