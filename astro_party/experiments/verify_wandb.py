import json
from pathlib import Path
import wandb

api=wandb.Api(timeout=30)
entity=api.default_entity
run=api.run(f'{entity}/astro-party-hw3/6o6gk5el')
history=run.history(samples=200,pandas=False)
keys={key for row in history for key in row}
required={'eval/rusher_win_rate','eval/sniper_win_rate','rollout/ep_rew_mean','train/value_loss'}
assert required.issubset(keys),(required-keys)
result=dict(verified=True,entity=entity,project='astro-party-hw3',run_id=run.id,
            run_url=run.url,project_url=run.url.rsplit('/runs/',1)[0],
            remote_state=run.state,remote_history_rows=len(history),verified_metrics=sorted(required))
Path('results/wandb_sync.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
