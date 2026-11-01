import json
from pathlib import Path
import wandb

api=wandb.Api(timeout=30)
entity=api.default_entity
keys={'global_step','motion/ship_speed_mean','motion/speed_mean','motion/velocity_reward_mean',
      'motion/covered_mean','motion/hunt_mode_mean','eval/rusher_win_rate','eval/sniper_win_rate','train/value_loss'}
data={}
for label,run_id,coefficient in [('velocity','xdfwkau6',.0015),('no_velocity','4r0t3h22',0.)]:
    run=api.run(f'{entity}/astro-party-hw3/{run_id}')
    rows=run.history(samples=300,pandas=False)
    present={k for row in rows for k in row}
    assert {'motion/ship_speed_mean','motion/velocity_reward_mean','train/value_loss'}.issubset(present)
    assert run.state=='finished'
    assert run.config['velocity_coef']==coefficient
    assert run.config['hunt_approach_coef']==6
    data[label]=dict(id=run_id,url=run.url,state=run.state,
        config={k:run.config[k] for k in ['velocity_coef','reward_profile','hunt_approach_coef','controller','seed']},
        history=[{k:v for k,v in row.items() if k in keys} for row in rows])
comparison=api.run(f'{entity}/astro-party-hw3/x4hdfhjo')
assert comparison.state=='finished'
assert comparison.summary['fast_velocity/sniper/win_rate']==.8
data['comparison']=dict(id=comparison.id,url=comparison.url,state=comparison.state)
Path('results/fast_wandb_history.json').write_text(json.dumps(data,indent=2))
print(json.dumps({k:{a:b for a,b in v.items() if a!='history'} for k,v in data.items()}),flush=True)
