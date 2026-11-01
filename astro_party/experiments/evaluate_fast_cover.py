"""Frozen-policy comparison: old cover, fast cover without/with speed bonus.

Only aggregate metrics go to W&B. Episode records and traces stay local.
"""
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
from train_astro_party import CoverBot,FastCoverBot


def load_finished(path,steps=40960):
    deadline=time.monotonic()+2400
    while True:
        try:
            data=Path(path).read_bytes()
            model=PPO.load(io.BytesIO(data),device='cpu')
            if model.num_timesteps>=steps:
                return model,hashlib.sha256(data).hexdigest()
        except (OSError,ValueError,EOFError,zipfile.BadZipFile):
            pass
        if time.monotonic()>deadline: raise RuntimeError('Checkpoint did not finish: '+path)
        print('Waiting for completed checkpoint:',path,flush=True)
        time.sleep(10)


def main():
    torch.set_num_threads(1)
    episodes,seed_base=100,65000
    run=wandb.init(project='astro-party-hw3',name='fast-cover-speed-ablation',
        mode='online',job_type='evaluation',config=dict(episodes=episodes,seed_base=seed_base),dir='.')
    result=dict(url=run.url,episodes=episodes,seed_base=seed_base,results={})
    paths=[('old_cover','models/model_astro_party_cover.zip'),
        ('fast_no_velocity','models/model_astro_party_fast_cover_novelo.zip'),
        ('fast_velocity','models/model_astro_party_fast_cover.zip')]
    try:
        for label,path in paths:
            model,digest=load_finished(path)
            result['results'][label]=dict(sha256=digest,steps=model.num_timesteps,opponents={})
            for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
                env=AstroPartyEvalEnv(); games=[]
                for seed in range(seed_base,seed_base+episodes):
                    obs,_=env.reset(seed=seed)
                    steps=covered=ship_steps=ship_speed=0
                    retreat_steps=retreat_covered=retreat_speed=0
                    empty_steps=empty_covered=direction_cases=direction_matches=0
                    enemy_ejections=enemy_respawns=0; pilot_start=None; pilot_finish=None
                    first_shot=None; trace=[]
                    while True:
                        s=State(obs['ship1']); hidden=CoverBot.planet_blocks(s)
                        active=bool(np.any(obs['ship1'][[48,53,58,63]]>.5))
                        speed=float(np.hypot(s.vx,s.vy))
                        covered+=int(hidden)
                        if not s.me_astro:
                            ship_steps+=1; ship_speed+=speed
                            if s.ammo==0 and not s.laser:
                                empty_steps+=1; empty_covered+=int(hidden)
                            if (s.ammo==0 or active or not s.ready) and not s.opp_astro:
                                retreat_steps+=1; retreat_covered+=int(hidden); retreat_speed+=speed
                                omega,_=FastCoverBot.orbit_motion(s)
                                ours=((s.px-600)*s.vy-(s.py-400)*s.vx)/max((s.px-600)**2+(s.py-400)**2,1.)
                                if abs(omega)>.025 and abs(ours)>.025:
                                    direction_cases+=1; direction_matches+=int(omega*ours>0)
                        action,_=model.predict(obs['ship1'],deterministic=True)
                        if seed==seed_base and steps%5==0:
                            trace.append(dict(t=steps/50,x=float(s.px),y=float(s.py),
                                enemy_x=float(s.px+s.dx),enemy_y=float(s.py+s.dy),covered=hidden,
                                ammo=s.ammo,speed=speed,enemy_pilot=bool(s.opp_astro),action=action.tolist()))
                        old_ammo=env.ships['ship1'].ammo
                        was_pilot=env.ships['ship2'].mode=='astronaut'
                        obs,_,term,trunc,_=env.step({'ship1':decode_action(action),
                            'ship2':decode_action(opponent(obs['ship2']))})
                        steps+=1
                        is_pilot=env.ships['ship2'].mode=='astronaut'
                        if is_pilot and not was_pilot: enemy_ejections+=1; pilot_start=steps
                        if was_pilot and not is_pilot: enemy_respawns+=1; pilot_start=None
                        if first_shot is None and env.ships['ship1'].ammo<old_ammo: first_shot=steps/50
                        if term['ship1'] or trunc['ship1']: break
                    win=env.ships['ship1'].alive and not env.ships['ship2'].alive
                    loss=not env.ships['ship1'].alive
                    ram_win=bool(win and np.hypot(*(env.ships['ship1'].pos-env.ships['ship2'].pos))<25.001)
                    if win and pilot_start is not None: pilot_finish=(steps-pilot_start)/50
                    games.append(dict(seed=seed,win=bool(win),loss=bool(loss),draw=not win and not loss,
                        steps=steps,ship_steps=ship_steps,ship_speed_sum=ship_speed,
                        ship_speed_mean=ship_speed/max(ship_steps,1),cover_fraction=covered/steps,
                        retreat_steps=retreat_steps,retreat_covered=retreat_covered,retreat_speed_sum=retreat_speed,
                        empty_steps=empty_steps,empty_covered=empty_covered,
                        direction_cases=direction_cases,direction_matches=direction_matches,
                        enemy_ejections=enemy_ejections,enemy_respawns=enemy_respawns,
                        pilot_finish_seconds=pilot_finish,ram_win=ram_win,first_shot_seconds=first_shot))
                    if trace:
                        Path(f'results/{label}_{name}_trace.json').write_text(json.dumps(trace,indent=2))
                    if len(games)%20==0: print(label,name,len(games),sum(g['win'] for g in games),'wins',flush=True)
                env.close()
                total=lambda key:sum(g[key] for g in games)
                finishes=[g['pilot_finish_seconds'] for g in games if g['pilot_finish_seconds'] is not None]
                shots=[g['first_shot_seconds'] for g in games if g['first_shot_seconds'] is not None]
                metrics=dict(win_rate=total('win')/episodes,loss_rate=total('loss')/episodes,
                    draw_rate=total('draw')/episodes,ship_speed_mean=total('ship_speed_sum')/max(total('ship_steps'),1),
                    retreat_speed_mean=total('retreat_speed_sum')/max(total('retreat_steps'),1),
                    retreat_cover_fraction=total('retreat_covered')/max(total('retreat_steps'),1),
                    empty_ammo_cover_fraction=total('empty_covered')/max(total('empty_steps'),1),
                    orbit_direction_match=total('direction_matches')/max(total('direction_cases'),1),
                    pilot_finish_rate=total('win')/max(total('enemy_ejections'),1),
                    enemy_respawns_mean=total('enemy_respawns')/episodes,
                    ram_win_rate=total('ram_win')/episodes,mean_steps=total('steps')/episodes,
                    pilot_finish_median=float(np.median(finishes)) if finishes else None,
                    first_shot_median=float(np.median(shots)) if shots else None)
                result['results'][label]['opponents'][name]=dict(metrics=metrics,games=games)
                run.log({f'{label}/{name}/{k}':v for k,v in metrics.items() if v is not None})
                Path('results/fast_cover_comparison.json').write_text(json.dumps(result,indent=2))
                print(label,name,metrics,flush=True)
    finally:
        run.finish()


if __name__=='__main__': main()
