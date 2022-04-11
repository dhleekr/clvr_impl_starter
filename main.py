import sprites_env
import gym
import torch
from ppo import PPO
import argparse
import sys
from eval_policy import eval_policy

from ppo_networks import CNN, Representation, Oracle

def train(env, baseline_class, hyperparameters, actor, critic, env_type):
    print(f"Training", flush=True)

    model = PPO(env=env, baseline_class=baseline_class, env_type=env_type, **hyperparameters)

    if actor != '' and critic != '':
        print(f"Loading in {actor} and {critic}...", flush=True)
        model.actor.load_state_dict(torch.load(actor))
        model.critic.load_state_dict(torch.load(critic))
        print(f"Successfully loaded.", flush=True)
    else:
        print(f"Training.", flush=True)

    model.learn(total_timesteps=3500000)

def main(args):
    hyperparameters = {
				'timesteps_per_batch': 2000, 
				'max_timesteps_per_episode': 40,
                'n_updates_per_iteration' : 10,
				'gamma': 0.99, 
				'lr': 3e-4, 
				'clip': 0.2,
                'value_loss_coef': 0.5,
                'entropy_coef': 0.01,
				'render': False,
				'render_every_i': 10
			  }

    # for v in ['v2']:
        # args.env_type = v
    if args.mode == 'Oracle':
        env = gym.make(f'SpritesState-{args.env_type}')
    else:
        env = gym.make(f'Sprites-{args.env_type}')

    train(env=env, baseline_class=args.mode, hyperparameters=hyperparameters, actor=args.actor_model, critic=args.critic_model, env_type=args.env_type)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='Oracle')             
    parser.add_argument('--env_type', dest='env_type', type=str, default='v0')             
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')     
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')   

    args = parser.parse_args()

    main(args)
