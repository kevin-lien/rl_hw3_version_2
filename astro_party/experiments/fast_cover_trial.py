"""Teacher-only movement, cover, pilot-chase and dodge diagnosis."""
import argparse
import json
from pathlib import Path
import numpy as np
from train_astro_party import FastCoverBot
from models.sniper_bot import State,_sim_min_dist,sniper_bot_action
from models.rusher_bot import rusher_bot_action
from envs.eval_astro_party_env import AstroPartyEvalEnv
from envs.game import decode_action


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed-base',type=int,default=64000)
    p.add_argument('--orbit-radius',type=float,default=220.)
    p.add_argument('--cruise-speed',type=float,default=240.)
    p.add_argument('--tag',default='fast_cover_v1')
    a=p.parse_args()
    bot=FastCoverBot(500,700,.8,60,28,orbit_radius=a.orbit_radius,cruise_speed=a.cruise_speed)
    off=FastCoverBot(500,700,.8,60,28,False,orbit_radius=a.orbit_radius,cruise_speed=a.cruise_speed)
    data=dict(config=vars(a),results={},dodge_checks=[])
    for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
        env=AstroPartyEvalEnv(); games=[]
        for seed in range(a.seed_base,a.seed_base+a.episodes):
            obs,_=env.reset(seed=seed)
            steps=speed=covered=hide_steps=hide_covered=direction_cases=direction_matches=0
            first_shot=None; pilot_start=None; pilot_kill_seconds=None; respawns=0
            while True:
                s=State(obs['ship1']); hidden=bot.planet_blocks(s)
                own_flight=bool(np.any(obs['ship1'][[48,53,58,63]]>.5))
                phase=bot.plan(s,own_flight)[0]
                speed+=np.hypot(s.vx,s.vy); covered+=int(hidden)
                if phase!='peek' and not s.me_astro and not s.opp_astro:
                    hide_steps+=1; hide_covered+=int(hidden)
                    omega,_=bot.orbit_motion(s)
                    ours=((s.px-600)*s.vy-(s.py-400)*s.vx)/max((s.px-600)**2+(s.py-400)**2,1.)
                    if abs(omega)>.025 and abs(ours)>.025:
                        direction_cases+=1; direction_matches+=int(ours*omega>0)
                action=bot(obs['ship1'])
                if s.opp_bullets and len(data['dodge_checks'])<200:
                    before=off(obs['ship1'])
                    if not np.array_equal(action[:2],before[:2]):
                        d0=_sim_min_dist(s,*before[:2],s.opp_bullets,60,s.me_astro)
                        d1=_sim_min_dist(s,*action[:2],s.opp_bullets,60,s.me_astro)
                        assert d1+1e-6>=d0
                        data['dodge_checks'].append(dict(before=d0,after=d1))
                ammo=env.ships['ship1'].ammo
                was_pilot=env.ships['ship2'].mode=='astronaut'
                obs,_,term,trunc,_=env.step({'ship1':decode_action(action),
                    'ship2':decode_action(opponent(obs['ship2']))})
                steps+=1
                is_pilot=env.ships['ship2'].mode=='astronaut'
                if not was_pilot and is_pilot: pilot_start=steps
                if was_pilot and not is_pilot: respawns+=1; pilot_start=None
                if first_shot is None and env.ships['ship1'].ammo<ammo: first_shot=steps/50
                if term['ship1'] or trunc['ship1']: break
            win=env.ships['ship1'].alive and not env.ships['ship2'].alive
            if win and pilot_start is not None: pilot_kill_seconds=(steps-pilot_start)/50
            loss=not env.ships['ship1'].alive
            games.append(dict(seed=seed,win=bool(win),draw=not win and not loss,steps=steps,
                mean_speed=float(speed/steps),cover_fraction=covered/steps,
                reload_cover_fraction=hide_covered/max(hide_steps,1),
                orbit_direction_match=direction_matches/max(direction_cases,1),
                first_shot=first_shot,pilot_kill_seconds=pilot_kill_seconds,enemy_respawns=respawns))
            if len(games)%5==0: print(name,len(games),sum(g['win'] for g in games),'wins',flush=True)
        env.close()
        summary={key:float(np.mean([g[key] for g in games])) for key in
            ['win','draw','steps','mean_speed','cover_fraction','reload_cover_fraction','orbit_direction_match','enemy_respawns']}
        data['results'][name]=dict(summary=summary,games=games)
        Path(f'results/{a.tag}.json').write_text(json.dumps(data,indent=2))
        print(name,summary,flush=True)


if __name__=='__main__': main()
