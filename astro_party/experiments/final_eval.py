"""Record every duel using the UNMODIFIED official loader and play function."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import torch
import wandb
from eval_astro_party import load_model,play
from models.rusher_bot import rusher_bot_action
from models.sniper_bot import sniper_bot_action


def wilson(wins,n):
    z=1.959963984540054
    p=wins/n
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half,center+half]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='models/model_astro_party.zip')
    p.add_argument('--seed-base',type=int,default=0)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--tag',default='public')
    p.add_argument('--wandb-mode',default='offline')
    a=p.parse_args()
    torch.set_num_threads(1)
    model=load_model(a.model)
    run=wandb.init(project='astro-party-hw3',name='final-'+a.tag,job_type='evaluation',
                   mode=a.wandb_mode,config=vars(a),dir='.')
    results=dict(model=a.model,sha256=hashlib.sha256(Path(a.model).read_bytes()).hexdigest(),
                 seed_base=a.seed_base,episodes=a.episodes,results={})
    table=wandb.Table(columns=['opponent','seed','win','steps'])
    try:
        for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
            rows=[]
            for seed in range(a.seed_base,a.seed_base+a.episodes):
                win,steps=play(model,opponent,seed)
                rows.append(dict(seed=seed,win=bool(win),steps=steps))
                table.add_data(name,seed,bool(win),steps)
                if len(rows)%20==0:
                    print(name,len(rows),sum(r['win'] for r in rows),flush=True)
            wins=sum(r['win'] for r in rows)
            results['results'][name]=dict(wins=wins,win_rate=wins/a.episodes,
                wilson95=wilson(wins,a.episodes),rows=rows)
            run.summary[name+'_win_rate']=wins/a.episodes
            Path(f'results/final_{a.tag}.json').write_text(json.dumps(results,indent=2))
        run.log({'games':table})
        artifact=wandb.Artifact('evaluation-'+a.tag,type='evaluation')
        artifact.add_file(f'results/final_{a.tag}.json')
        run.log_artifact(artifact)
        print(json.dumps({k:v['win_rate'] for k,v in results['results'].items()}),flush=True)
    finally:
        run.finish()


if __name__=='__main__':
    main()
