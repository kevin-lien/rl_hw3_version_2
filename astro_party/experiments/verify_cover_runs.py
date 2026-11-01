import json
from pathlib import Path
import wandb

api=wandb.Api(timeout=30)
entity=api.default_entity
items=[]
reproduction=json.loads(Path('results/reproduction_check.json').read_text())
reproduction_run=reproduction.get('run_id','sueyklu0')
for run_id,keys in [(reproduction_run,{'eval/sniper_win_rate','train/value_loss'}),
                    ('cj240jwb',{'eval/sniper_win_rate','train/value_loss'}),
                    ('em6va3hq',{'cover/sniper/win_rate','reference/sniper/win_rate'})]:
    run=api.run(f'{entity}/astro-party-hw3/{run_id}')
    rows=run.history(samples=300,pandas=False)
    present={key for row in rows for key in row}
    assert keys.issubset(present),(run_id,keys-present)
    assert run.state=='finished',(run_id,run.state)
    items.append(dict(id=run_id,url=run.url,state=run.state,rows=len(rows),verified_metrics=sorted(keys)))
Path('results/cover_wandb_verified.json').write_text(json.dumps(items,indent=2))
print(json.dumps(items),flush=True)
