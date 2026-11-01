"""Regression and controlled-state checks for the optional reward profile."""
from pathlib import Path
import numpy as np
from envs.train_astro_party_env import AstroPartyTrainEnv,VELOCITY_REWARD_COEF,STEP_PENALTY
from models.sniper_bot import sniper_bot_action

old_namespace={}
exec(compile(Path('original/train_env_before_fast_cover.py').read_text(),'<old-env>','exec'),old_namespace)
OldEnv=old_namespace['AstroPartyTrainEnv']
old=OldEnv(opponent_fn=sniper_bot_action)
new=AstroPartyTrainEnv(opponent_fn=sniper_bot_action)
left,_=old.reset(seed=73000); right,_=new.reset(seed=73000)
rng=np.random.default_rng(12)
for step in range(1000):
    action=rng.uniform(-1,1,3).astype(np.float32)
    left,r0,t0,tr0,_=old.step(action)
    right,r1,t1,tr1,_=new.step(action)
    assert np.array_equal(left,right) and r0==r1 and t0==t1 and tr0==tr1,(step,r0,r1)
    if t0 or tr0:
        left,_=old.reset(seed=73001+step); right,_=new.reset(seed=73001+step)
old.close();new.close()
env=AstroPartyTrainEnv(reward_profile='fast_cover')
env.reset(seed=1)
me,enemy=env.game.ships['ship1'],env.game.ships['ship2']
me.pos[:]=[380,400]; enemy.pos[:]=[950,400]
me.vel[:]=[0,0]; env.prev_dist=env.distance(); stationary=env.reward([])
me.vel[:]=[0,260]; env.prev_dist=env.distance(); moving=env.reward([])
assert moving>stationary
assert moving<0  # Speed alone cannot create positive per-step return.
enemy.mode='astronaut'; me.vel[:]=[260,0]; env.prev_dist=env.distance()+.01
hunting=env.reward([])
assert env.last_reward_parts['approach_coefficient']==6.
assert hunting>.05
me.vel[:]=[-260,0]; env.prev_dist=env.distance(); env.reward([])
assert env.last_reward_parts['velocity_reward']==0  # Away from pilot is not rewarded.
assert 1500*VELOCITY_REWARD_COEF<20
env.close()
print('1000-step original-reward regression passed; speed bonus bounded; pilot coefficient 6; no reward for moving away from pilot.')
