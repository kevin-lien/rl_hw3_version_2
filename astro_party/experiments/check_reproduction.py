"""Check tensor equality, not ZIP byte equality (source metadata can change)."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO

p=argparse.ArgumentParser()
p.add_argument('--reproduced',default='reproduce_check/models/model_astro_party.zip')
p.add_argument('--output',default='results/reproduction_check.json')
a=p.parse_args()
torch.set_num_threads(1)
reference=PPO.load('models/model_astro_party_defense_reference.zip',device='cpu')
reproduced=PPO.load(a.reproduced,device='cpu')
left,right=reference.policy.state_dict(),reproduced.policy.state_dict()
assert left.keys()==right.keys()
differences={key:float((left[key]-right[key]).abs().max()) for key in left}
equal=all(torch.equal(left[key],right[key]) for key in left)
result=dict(command='python train_astro_party.py',uses_prior_checkpoint=False,
    reproduced_path=a.reproduced,reference_steps=reference.num_timesteps,reproduced_steps=reproduced.num_timesteps,
    parameter_count=sum(p.numel() for p in reproduced.policy.parameters()),all_policy_tensors_equal=equal,
    maximum_absolute_difference=max(differences.values()),tensor_differences=differences,
    source_layout='flat train_astro_party.py and train_astro_party_env.py; stock envs and baselines',
    reference_sha256=hashlib.sha256(Path('models/model_astro_party_defense_reference.zip').read_bytes()).hexdigest(),
    reproduced_sha256=hashlib.sha256(Path(a.reproduced).read_bytes()).hexdigest())
source_root=Path(a.reproduced).parents[1]
result['training_source_sha256']=hashlib.sha256((source_root/'train_astro_party.py').read_bytes()).hexdigest()
result['training_env_sha256']=hashlib.sha256((source_root/'train_astro_party_env.py').read_bytes()).hexdigest()
run_infos=list(source_root.glob('wandb/run-*/files/run_info.json'))+list(source_root.glob('wandb/offline-run-*/files/run_info.json'))
if run_infos:
    info=json.loads(max(run_infos,key=lambda path:path.stat().st_mtime).read_text())
    result['run_id']=info['id']
    result['run_url']=info['url']
Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
