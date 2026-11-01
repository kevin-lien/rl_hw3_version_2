"""Preserve the trained-policy dodge comparison for the fast-cover model."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from envs.eval_astro_party_env import AstroPartyEvalEnv
from envs.game import decode_action
from models.sniper_bot import State,_sim_min_dist,sniper_bot_action

torch.set_num_threads(1)
model=PPO.load('models/model_astro_party_fast_cover.zip',device='cpu')
off=PPO.load('models/model_astro_party_fast_cover.zip',device='cpu')
off.policy.features_extractor.teacher.dodge=False
off.policy.features_extractor._feature_cache.clear()
checks=[]
for seed in range(67000,67010):
    env=AstroPartyEvalEnv(); obs,_=env.reset(seed=seed)
    while True:
        action,_=model.predict(obs['ship1'],deterministic=True)
        s=State(obs['ship1'])
        if s.opp_bullets:
            baseline,_=off.predict(obs['ship1'],deterministic=True)
            if not np.allclose(action[:2],baseline[:2],atol=1e-5):
                before=_sim_min_dist(s,*baseline[:2],s.opp_bullets,60,s.me_astro)
                after=_sim_min_dist(s,*action[:2],s.opp_bullets,60,s.me_astro)
                checks.append(dict(before=before,after=after,improvement=after-before))
        if len(checks)>=200: break
        obs,_,term,trunc,_=env.step({'ship1':decode_action(action),
            'ship2':decode_action(sniper_bot_action(obs['ship2']))})
        if term['ship1'] or trunc['ship1']: break
    env.close()
    if len(checks)>=200: break
result=dict(model_steps=model.num_timesteps,checks=len(checks),
    improved=sum(r['improvement']>0 for r in checks),
    minimum_improvement=min(r['improvement'] for r in checks),
    median_improvement=float(np.median([r['improvement'] for r in checks])),rows=checks)
Path('results/fast_trained_dodge_check.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
