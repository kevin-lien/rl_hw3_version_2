import argparse
import json
from pathlib import Path
from experiments.physics_defense import PhysicsBot
from experiments.probe import play
from models.sniper_bot import sniper_bot_action
from models.rusher_bot import rusher_bot_action

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['blocking','collisions','both'],default='blocking')
p.add_argument('--seed-base',type=int,default=10000)
p.add_argument('--episodes',type=int,default=100)
a=p.parse_args()
bot=PhysicsBot(collisions=a.mode!='blocking',blocking=a.mode!='collisions',
               horizon=60,margin=28,lead=.8,near=500,far=700)
data=dict(config=vars(a),results={})
for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
    games=[]
    for seed in range(a.seed_base,a.seed_base+a.episodes):
        games.append(play(bot,opponent,seed))
        if len(games)%20==0: print(name,len(games),sum(g[0] for g in games),flush=True)
    data['results'][name]=dict(wins=sum(g[0] for g in games),games=games)
    Path(f'results/physics_{a.mode}_{a.seed_base}.json').write_text(json.dumps(data,indent=2))
