import argparse
import json
from pathlib import Path
from experiments.defense import DefenseBot
from experiments.probe import play
from models.sniper_bot import sniper_bot_action
from models.rusher_bot import rusher_bot_action

p=argparse.ArgumentParser()
p.add_argument('--horizon',type=int,default=25)
p.add_argument('--margin',type=float,default=28)
p.add_argument('--near',type=float,default=500)
p.add_argument('--far',type=float,default=700)
p.add_argument('--lead',type=float,default=.8)
p.add_argument('--episodes',type=int,default=100)
p.add_argument('--seed-base',type=int,default=10000)
a=p.parse_args()
bot=DefenseBot(horizon=a.horizon,margin=a.margin,near=a.near,far=a.far,lead=a.lead)
results=dict(config=vars(a),results={})
for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
    games=[]
    for s in range(a.seed_base,a.seed_base+a.episodes):
        games.append(play(bot,opponent,s))
        if len(games)%20==0:
            print(name,len(games),sum(g[0] for g in games),flush=True)
    results['results'][name]=dict(wins=sum(g[0] for g in games),losses=sum(g[1] for g in games),games=games)
    suffix='' if a.seed_base==10000 else f'_seed{a.seed_base}'
    Path(f'results/teacher_h{a.horizon}_m{a.margin}_n{a.near}_l{a.lead}{suffix}.json').write_text(json.dumps(results,indent=2))
