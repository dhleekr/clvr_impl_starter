# CLVR Implementation Project - Starter Code

## Reference
The overall dataset and environment code, including ['sprites_datagen'](sprites_datagen) and ['sprites_env'](sprites_env), cloned from https://github.com/kpertsch/clvr_impl_starter.

## Baselines
- CNN
- Image-scratch
- Image-reconstruction or Image-reconstruction-finetune
- Reward-prediction or Reward-prediction-finetune
- Oracle

## Env type
- v0 (0 distractor)
- v1 (1 distractor)
- v2 (2 distractors)

## Example Commands
To train a reward-induced representation, run:
'''
python reward_prediction.py
'''

For training a PPO agent using diverse baselines, run:
'''
python main.py --mode {baseline name} --env_type {env type}
'''
