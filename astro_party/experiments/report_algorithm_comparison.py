"""Matched short-budget PPO/A2C comparison from the same BC initialization."""
import json
from pathlib import Path
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from train_astro_party import make_env,evaluate

OUT=Path('../report_work/results')
torch.set_num_threads(1)
base=PPO.load('models/6o6gk5el_bc.zip',device='cpu')
run=wandb.init(project='astro-party-hw3',name='report-ppo-a2c-matched',mode='offline',
    config=dict(budget=8192,rollout_seeds=[42,43,44],eval_seed=72000,eval_episodes=30,
                learning_rate=1e-6,initialization='same 6o6gk5el BC checkpoint',gamma=.995,
                note='Short-budget warm-start comparison, not complete independent training seeds'))
rows=[]
for seed in [42,43,44]:
    for label,cls in [('PPO',PPO),('A2C',A2C)]:
        env=DummyVecEnv([make_env for _ in range(4)]);env.seed(100000+seed)
        kwargs=dict(policy_kwargs=base.policy_kwargs,device='cpu',seed=seed,
            learning_rate=1e-6,n_steps=512,gamma=.995,gae_lambda=.95,ent_coef=0.,verbose=0)
        if label=='PPO':kwargs.update(batch_size=256,n_epochs=5,clip_range=.1,target_kl=.02)
        else:kwargs.update(normalize_advantage=True,use_rms_prop=False)
        model=cls('MlpPolicy',env,**kwargs)
        model.policy.load_state_dict(base.policy.state_dict())
        assert all(torch.equal(v,base.policy.state_dict()[k]) for k,v in model.policy.state_dict().items())
        model.learn(8192)
        metrics,episodes=evaluate(model,30,72000)
        row=dict(algorithm=label,seed=seed,steps=model.num_timesteps,metrics=metrics,episodes=episodes)
        rows.append(row)
        run.log(dict(algorithm=label,rollout_seed=seed,**metrics))
        (OUT/'astro_algorithm_comparison.json').write_text(json.dumps(rows,indent=2))
        print(label,seed,json.dumps(metrics),flush=True)
        env.close()
run.finish()
