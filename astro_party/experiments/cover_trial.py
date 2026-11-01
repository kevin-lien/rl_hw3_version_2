"""Diagnose cover behavior from public observations, before training a model."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from train_astro_party import CoverBot,DefenseBot
from models.sniper_bot import State,_sim_min_dist,sniper_bot_action
from models.rusher_bot import rusher_bot_action
from envs.eval_astro_party_env import AstroPartyEvalEnv
from envs.game import decode_action


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed-base',type=int,default=60000)
    p.add_argument('--radius',type=float,default=180)
    p.add_argument('--peek-angle',type=float,default=1.05)
    p.add_argument('--opening-speed',type=float,default=80)
    p.add_argument('--tag',default='cover_v1')
    a=p.parse_args()
    bot=CoverBot(500,700,.8,60,28,cover_radius=a.radius,peek_angle=a.peek_angle,opening_speed=a.opening_speed)
    no_dodge=CoverBot(500,700,.8,60,28,False,cover_radius=a.radius,peek_angle=a.peek_angle,opening_speed=a.opening_speed)
    results=dict(config=vars(a),results={},dodge_checks=[])
    for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
        games=[]
        env=AstroPartyEvalEnv()
        for seed in range(a.seed_base,a.seed_base+a.episodes):
            obs,_=env.reset(seed=seed)
            steps=covered=early_covered=early_steps=0
            first_shot=None
            phases={}
            trace=[]
            while True:
                s=State(obs['ship1'])
                phase=bot.plan(s)[0]
                if s.me_astro: phase='pilot'
                elif s.opp_astro: phase='hunt'
                elif np.hypot(s.dx,s.dy)<280 and not bot.planet_blocks(s): phase='survival'
                phases[phase]=phases.get(phase,0)+1
                hidden=bot.planet_blocks(s)
                covered+=int(hidden)
                if steps<250:
                    early_covered+=int(hidden); early_steps+=1
                action=bot(obs['ship1'])
                if s.opp_bullets and len(results['dodge_checks'])<300:
                    before=no_dodge(obs['ship1'])
                    if not np.array_equal(before[:2],action[:2]):
                        d0=_sim_min_dist(s,*before[:2],s.opp_bullets,60,s.me_astro)
                        d1=_sim_min_dist(s,*action[:2],s.opp_bullets,60,s.me_astro)
                        results['dodge_checks'].append(dict(before=d0,after=d1,improvement=d1-d0))
                        assert d1+1e-6>=d0,(seed,steps,d0,d1)
                if seed==a.seed_base and steps%5==0:
                    trace.append(dict(t=steps/50,x=float(s.px),y=float(s.py),
                        enemy_x=float(s.px+s.dx),enemy_y=float(s.py+s.dy),phase=phase,
                        covered=hidden,ammo=s.ammo,action=action.tolist()))
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
                steps=steps,first_shot_seconds=first_shot,cover_fraction=covered/steps,
                opening_cover_fraction=early_covered/early_steps,phases=phases))
            if trace:
                Path(f'results/{a.tag}_{name}_trace.json').write_text(json.dumps(trace,indent=2))
            if len(games)%5==0:
                print(name,len(games),sum(g['win'] for g in games),'wins',flush=True)
        env.close()
        shots=[g['first_shot_seconds'] for g in games if g['first_shot_seconds'] is not None]
        result=dict(win_rate=sum(g['win'] for g in games)/len(games),
            draw_rate=sum(g['draw'] for g in games)/len(games),
            opening_cover_fraction=float(np.mean([g['opening_cover_fraction'] for g in games])),
            first_shot_median=float(np.median(shots)) if shots else None,games=games)
        results['results'][name]=result
        print(name,{k:v for k,v in result.items() if k!='games'},flush=True)
        Path(f'results/{a.tag}.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__': main()
