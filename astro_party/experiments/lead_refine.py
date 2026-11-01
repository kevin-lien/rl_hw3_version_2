import json
from pathlib import Path
from experiments.defense import DefenseBot
from experiments.probe import play
from models.sniper_bot import sniper_bot_action
from models.rusher_bot import rusher_bot_action

results=[]
for lead in [.82,.85,.88]:
    bot=DefenseBot(horizon=60,margin=28,near=500,far=700,lead=lead)
    row=dict(lead=lead)
    for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
        games=[play(bot,opponent,s) for s in range(10000,10060)]
        row[name]=sum(g[0] for g in games)/len(games)
        print(row,flush=True)
    results.append(row)
    Path('results/lead_refine.json').write_text(json.dumps(results,indent=2))
