import json
from pathlib import Path
from experiments.advanced import AdvancedBot
from experiments.probe import play
from models.sniper_bot import sniper_bot_action
from models.rusher_bot import rusher_bot_action

results=[]
for near,far in [(380,620),(500,700)]:
    bot=AdvancedBot(horizon=30,lead=.8,near=near,far=far)
    row=dict(near=near,far=far,horizon=30)
    for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
        games=[play(bot,opponent,s) for s in range(10000,10100)]
        row[name]=sum(g[0] for g in games)/len(games)
        print(row,flush=True)
    results.append(row)
    Path('results/advanced_validation.json').write_text(json.dumps(results,indent=2))
