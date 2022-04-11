from ppo_networks import ActorCritic
import torch
import torch.nn as nn
import gym
import cv2
import sprites_env


env = gym.make('Sprites-v2')

print('Testing {actor_model}', flush=True)
policy = ActorCritic(env.observation_space, env.action_space, mode='Reward_prediction_finetune')
policy.load_state_dict(torch.load('./model/actor_critic_Reward_prediction_finetune_v0.pth'))

obs = env.reset()

for i in range(40):
    im = env.render()
    cv2.imwrite(f'./rollout/v2/{i+1}.png', im)

    with torch.no_grad():
        action, _, log_prob = policy(obs)
    action, log_prob = action.cpu().numpy(), log_prob.cpu().numpy()
    obs, rew, done, _ = env.step(action)
