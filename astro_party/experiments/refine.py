import argparse
import json
from pathlib import Path
from experiments.defense import DefenseBot
from experiments.probe import play
from models.sniper_bot import sniper_bot_action
from models.rusher_bot import rusher_bot_action

p=argparse.ArgumentParser()
p.add_argument('--horizon',type=int,default=60)
p.add_argument('--episodes',type=int,default=40)
a=p.parse_args()
results=[]
for near,far in [(380,620),(650,850),(500,700)]:
    for lead in [.7,.9,1.0]:
        bot=DefenseBot(horizon=a.horizon,margin=28,near=near,far=far,lead=lead)
        row=dict(horizon=a.horizon,near=near,far=far,lead=lead)
        for name,opp in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
            games=[play(bot,opp,s) for s in range(10000,10000+a.episodes)]
            row[name]=sum(g[0] for g in games)/len(games)
        print(row,flush=True)
        results.append(row)
        Path(f'results/refine{a.horizon}.json').write_text(json.dumps(results,indent=2))
