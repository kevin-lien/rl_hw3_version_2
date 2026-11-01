"""Training-side controller diagnosis; uses no public or holdout test seeds."""
import argparse
import json
from pathlib import Path
from experiments.probe import PursuitBot, play
from models.sniper_bot import RuleBot, sniper_bot_action
from models.rusher_bot import rusher_bot_action

p = argparse.ArgumentParser()
p.add_argument('--episodes',type=int,default=20)
p.add_argument('--seed-base',type=int,default=31000)
a = p.parse_args()
results = []
for near,far in [(250,400),(380,620),(500,700),(80,600),(0,900)]:
    for tol in [.6,1.0,1.8]:
        bot = PursuitBot(near,far,tol,True,1.2)
        row = dict(near=near,far=far,tol=tol)
        for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
            games = [play(bot,opponent,s) for s in range(a.seed_base,a.seed_base+a.episodes)]
            row[name] = sum(g[0] for g in games)/len(games)
        results.append(row)
        print(row,flush=True)
        Path('results/sweep.json').write_text(json.dumps(results,indent=2))
