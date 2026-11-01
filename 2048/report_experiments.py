"""Local report measurements. Run inside 2048; never used by grading."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from train_2048 import DoubleDQN
from envs.train_2048_env import My2048Env, stack
from envs.eval_2048_env import Eval2048Env

OUT=Path('../report_work/results')
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)

def evaluate(model):
    rows=[]
    for seed in range(100):
        env=Eval2048Env(); obs,_=env.reset(seed=seed)
        for step in range(3000):
            action,_=model.predict(obs,deterministic=True)
            obs,r,done,truncated,info=env.step(int(action))
            if done or truncated: break
        else: raise RuntimeError('Evaluation cap reached')
        rows.append(dict(seed=seed,highest=int(info['highest']),score=float(info['score']),steps=step+1))
    return dict(rows=rows,average_score=float(np.mean([r['score'] for r in rows])),
                counts={str(v):sum(r['highest']==v for r in rows) for v in sorted(set(r['highest'] for r in rows))},
                best=max(r['highest'] for r in rows))

def mc(model,label):
    states=[]
    # States come from each greedy policy's own trajectories, not random boards.
    for seed in range(71000,71030):
        env=My2048Env(); obs,_=env.reset(seed=seed)
        for step in range(1000):
            if step in [0,5,15,30] and len(states)<12:
                states.append((env.Matrix.copy(),env.score,env.foul_count))
            action,_=model.predict(obs,deterministic=True)
            obs,r,done,truncated,_=env.step(int(action))
            if done or truncated: break
        if len(states)>=12: break
    rows=[]
    for index,(board,score,fouls) in enumerate(states):
        observation=stack(board)
        with torch.no_grad():
            tensor,_=model.policy.obs_to_tensor(observation)
            q=model.q_net(tensor)[0].numpy()
        action=int(q.argmax())
        envs=[]
        # Independent future RNG streams keep 2/4 tile probability and uniform empty-cell spawn.
        for rollout in range(64):
            env=My2048Env();env.reset(seed=800000+index*1000+rollout)
            env.Matrix=board.copy();env.score=score;env.foul_count=fouls
            envs.append(env)
        returns=np.zeros(64);active=list(range(64));discount=1.;capped=False
        for step in range(2000):
            if not active: break
            if step==0: actions=np.full(len(active),action)
            else:
                actions,_=model.predict(np.stack([stack(envs[i].Matrix) for i in active]),deterministic=True)
            remaining=[]
            for i,a in zip(active,actions):
                _,r,done,truncated,_=envs[i].step(int(a))
                returns[i]+=discount*r
                if not(done or truncated):remaining.append(i)
            active=remaining;discount*=model.gamma
        if active:capped=True
        # Student t 0.975 quantile with 63 degrees of freedom.
        mean=float(returns.mean());half=float(1.998340543*returns.std(ddof=1)/8)
        rows.append(dict(state=index,action=action,predicted=float(q[action]),mc=mean,ci95_half=half,
                         capped=capped,board=board.tolist(),returns=returns.tolist()))
    result=dict(label=label,rollouts=64,gamma=model.gamma,states=rows,
                mean_bias=float(np.mean([r['predicted']-r['mc'] for r in rows])),
                mean_absolute_gap=float(np.mean([abs(r['predicted']-r['mc']) for r in rows])))
    (OUT/f'mc_{label}.json').write_text(json.dumps(result,indent=2))
    print(label,'MC',result['mean_bias'],flush=True)

def main():
    final=DQN.load('models/model_2048_weight1000.zip',device='cpu')
    metadata={key:getattr(final,key,None) for key in ['num_timesteps','double_q','batch_size','gamma','learning_rate','buffer_size','learning_starts','target_update_interval','seed']}
    metadata['parameter_count']=sum(p.numel() for p in final.q_net.parameters())
    metadata['evaluation']=evaluate(final)
    (OUT/'2048_final.json').write_text(json.dumps(metadata,indent=2))
    print('Final checkpoint:',metadata,flush=True)
    mc(final,'final_ddqn')
    comparison={}
    for label,double in [('dqn',False),('ddqn',True)]:
        env=DummyVecEnv([My2048Env for _ in range(4)])
        model=DoubleDQN('MlpPolicy',env,double_q=double,seed=42,device='cpu',
            buffer_size=100000,batch_size=256,policy_kwargs=dict(net_arch=[256,256]),
            exploration_fraction=.1,exploration_initial_eps=1.,exploration_final_eps=.05,
            tensorboard_log=str(OUT/'tb'),verbose=0)
        start=time.time();model.learn(40000,tb_log_name=label)
        model.save(str(OUT/f'comparison_{label}'))
        comparison[label]=dict(steps=model.num_timesteps,seconds=time.time()-start,evaluation=evaluate(model))
        (OUT/'2048_algorithm_comparison.json').write_text(json.dumps(comparison,indent=2))
        mc(model,label);env.close()
        print(label,'finished',flush=True)

if __name__=='__main__':main()
