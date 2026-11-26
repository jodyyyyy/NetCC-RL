"""Frozen existing SB3 formal settings; only phase feature extraction is new."""
import importlib.metadata as md
import json
from pathlib import Path
import numpy as np
from .sb3 import PhaseExtractor


def build(algorithm, env, seed, device):
    if md.version('stable-baselines3')!='2.4.1':raise ValueError('SB3 2.4.1 required')
    from stable_baselines3 import PPO,SAC,TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.monitor import Monitor
    spec=json.loads((Path(__file__).with_name('configs')/'formal-hyperparameters-v1.json').read_text())
    p=spec['algorithms'][algorithm]
    settings=dict(learning_rate=p['learning_rate'],gamma=p['gamma'],
                  policy_kwargs=dict(net_arch=p['net_arch'],features_extractor_class=PhaseExtractor))
    if algorithm=='ppo':
        settings.update(n_steps=p['rollout_steps'],batch_size=p['batch_size'],n_epochs=p['epochs'],
            gae_lambda=p['gae_lambda'],clip_range=p['clip_range'],ent_coef=p['entropy_coef'],
            vf_coef=p['value_coef'],max_grad_norm=p['max_grad_norm'],normalize_advantage=p['normalize_advantage'],target_kl=None)
    else:
        settings.update(buffer_size=p['replay_capacity'],learning_starts=p['learning_starts'],batch_size=p['batch_size'],
                        tau=p['tau'],train_freq=p['train_freq_env_steps'],gradient_steps=p['gradient_steps'])
        if algorithm=='sac':settings.update(ent_coef=p['entropy_coef'],target_entropy=p['target_entropy'])
        else:settings.update(policy_delay=p['policy_delay'],target_policy_noise=p['target_policy_noise'],
            target_noise_clip=p['target_noise_clip'],action_noise=NormalActionNoise(np.zeros(env.action_space.shape[0]),np.full(env.action_space.shape[0],p['exploration_noise']['sigma'])))
    return dict(ppo=PPO,sac=SAC,td3=TD3)[algorithm]('MlpPolicy',Monitor(env),seed=seed,device=device,verbose=0,**settings)
