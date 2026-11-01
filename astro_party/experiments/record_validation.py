import argparse
import hashlib
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from train_astro_party import evaluate

p=argparse.ArgumentParser()
p.add_argument('--model',default='models/model_astro_party.zip')
p.add_argument('--output',default='results/validation_final.json')
p.add_argument('--episodes',type=int,default=100)
p.add_argument('--seed-base',type=int,default=10000)
a=p.parse_args()
torch.set_num_threads(1)
model=PPO.load(a.model,device='cpu')
metrics,rows=evaluate(model,a.episodes,a.seed_base)
data=dict(metrics=metrics,episodes=rows,sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
          model_steps=model.num_timesteps,parameters=sum(p.numel() for p in model.policy.parameters()))
Path(a.output).write_text(json.dumps(data,indent=2))
print(json.dumps(data['metrics']),flush=True)
