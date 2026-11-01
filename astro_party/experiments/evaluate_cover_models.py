"""Paired evaluation of saved PPO policies; upload aggregate metrics only."""
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
import wandb
from stable_baselines3 import PPO
from envs.eval_astro_party_env import AstroPartyEvalEnv
from envs.game import decode_action
from models.sniper_bot import State,sniper_bot_action
from models.rusher_bot import rusher_bot_action
from train_astro_party import CoverBot


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed-base',type=int,default=61000)
    p.add_argument('--candidate',default='models/model_astro_party_cover.zip')
    p.add_argument('--reference',default='models/model_astro_party_defense_reference.zip')
    p.add_argument('--wandb-mode',default='online')
    p.add_argument('--expected-steps',type=int,default=40960)
    a=p.parse_args()
    torch.set_num_threads(1)
    run=wandb.init(project='astro-party-hw3',name='cover-paired-evaluation',
        job_type='evaluation',mode=a.wandb_mode,config=dict(episodes=a.episodes,seed_base=a.seed_base),dir='.')
    result=dict(config=vars(a),url=run.url,results={})
    try:
        for label,path in [('reference',a.reference),('cover',a.candidate)]:
            if label=='cover':
                deadline=time.monotonic()+1800
                while True:
                    try:
                        loaded_bytes=Path(path).read_bytes()
                        model=PPO.load(io.BytesIO(loaded_bytes),device='cpu')
                        if model.num_timesteps>=a.expected_steps: break
                    except (OSError,ValueError,EOFError,zipfile.BadZipFile):
                        pass
                    if time.monotonic()>deadline:
                        raise RuntimeError('Candidate did not finish the expected training budget')
                    print('Waiting for cover checkpoint:',a.expected_steps,'steps',flush=True)
                    time.sleep(10)
            else:
                loaded_bytes=Path(path).read_bytes()
                model=PPO.load(io.BytesIO(loaded_bytes),device='cpu')
            result['results'][label]=dict(sha256=hashlib.sha256(loaded_bytes).hexdigest(),
                model_steps=model.num_timesteps,parameters=sum(p.numel() for p in model.policy.parameters()),opponents={})
            for opponent_name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
                env=AstroPartyEvalEnv()
                games=[]
                for seed in range(a.seed_base,a.seed_base+a.episodes):
                    obs,_=env.reset(seed=seed)
                    steps=hidden=early_hidden=early_steps=0
                    first_shot=None
                    early_eject=False
                    trace=[]
                    while True:
                        s=State(obs['ship1'])
                        covered=CoverBot.planet_blocks(s)
                        hidden+=int(covered)
                        if steps<250:
                            early_hidden+=int(covered); early_steps+=1
                            early_eject=early_eject or s.me_astro
                        action,_=model.predict(obs['ship1'],deterministic=True)
                        if seed==a.seed_base and steps%5==0:
                            trace.append(dict(t=steps/50,x=float(s.px),y=float(s.py),
                                enemy_x=float(s.px+s.dx),enemy_y=float(s.py+s.dy),covered=covered,
                                ammo=s.ammo,action=action.tolist()))
                        ammo=env.ships['ship1'].ammo
                        obs,_,term,trunc,_=env.step({'ship1':decode_action(action),
                            'ship2':decode_action(opponent(obs['ship2']))})
                        steps+=1
                        if first_shot is None and env.ships['ship1'].ammo<ammo:
                            first_shot=steps/50
                        if term['ship1'] or trunc['ship1']: break
                    win=env.ships['ship1'].alive and not env.ships['ship2'].alive
                    loss=not env.ships['ship1'].alive
                    games.append(dict(seed=seed,win=bool(win),loss=bool(loss),draw=not win and not loss,
                        steps=steps,first_shot_seconds=first_shot,cover_fraction=hidden/steps,
                        opening_cover_fraction=early_hidden/early_steps,opening_eject=bool(early_eject)))
                    if trace:
                        Path(f'results/cover_model_{label}_{opponent_name}_trace.json').write_text(json.dumps(trace,indent=2))
                    if len(games)%20==0:
                        print(label,opponent_name,len(games),sum(g['win'] for g in games),'wins',flush=True)
                env.close()
                shots=[g['first_shot_seconds'] for g in games if g['first_shot_seconds'] is not None]
                metrics=dict(win_rate=float(np.mean([g['win'] for g in games])),
                    loss_rate=float(np.mean([g['loss'] for g in games])),draw_rate=float(np.mean([g['draw'] for g in games])),
                    mean_steps=float(np.mean([g['steps'] for g in games])),
                    opening_cover_fraction=float(np.mean([g['opening_cover_fraction'] for g in games])),
                    opening_eject_rate=float(np.mean([g['opening_eject'] for g in games])),
                    cover_fraction=float(np.mean([g['cover_fraction'] for g in games])),
                    first_shot_median=float(np.median(shots)) if shots else None)
                result['results'][label]['opponents'][opponent_name]=dict(metrics=metrics,games=games)
                run.log({f'{label}/{opponent_name}/{key}':value for key,value in metrics.items() if value is not None})
                Path('results/cover_model_comparison.json').write_text(json.dumps(result,indent=2))
                print(label,opponent_name,metrics,flush=True)
    finally:
        run.finish()


if __name__=='__main__': main()
