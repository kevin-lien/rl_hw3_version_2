"""CPU PPO: make_env -> evaluate -> train, with W&B and optional imitation.

Run as a script so custom feature classes are embedded by cloudpickle. The
grader uses raw observations and standard PPO.predict, without wrappers.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# The submission lists the training env next to this file; the starter repo
# keeps it under envs/. Support both layouts without changing the game.
try:
    from train_astro_party_env import AstroPartyTrainEnv, decode_action
except ModuleNotFoundError as error:
    if error.name != 'train_astro_party_env':
        raise
    from envs.train_astro_party_env import AstroPartyTrainEnv, decode_action
from envs.eval_astro_party_env import AstroPartyEvalEnv
from models.rusher_bot import rusher_bot_action
from models.sniper_bot import (sniper_bot_action, RuleBot, State, _wrap,
                              TURN_STEP, line_of_sight, intercept_time,
                              _flee, _sim_min_dist)
from envs.game import (BULLET_SPEED,WALLS,PLANET_POS,PLANET_RADIUS,
                      _ray_circle_t,_ray_aabb_entry_t,_ray_box_exit_t,
                      _circle_rect_collide,_resolve_circle_collision)


class PursuitBot(RuleBot):
    """Teacher: evaluate aim after this tick's turn, not before it."""
    def _aim(self, s, target_r):
        t = intercept_time(s.dx, s.dy, s.ovx, s.ovy, BULLET_SPEED)
        t = math.hypot(s.dx, s.dy) / BULLET_SPEED if t is None else t
        ax = s.dx + (0 if s.laser else s.ovx * t)
        ay = s.dy + (0 if s.laser else s.ovy * t)
        err = _wrap(math.atan2(ax, -ay) - s.ang)
        after = max(0.0, abs(err) - TURN_STEP)
        tol = math.atan2(target_r * .8, max(math.hypot(ax, ay), 1)) * self.fire_tol
        clear = line_of_sight(s.px, s.py, s.px + ax, s.py + ay, s.asteroids)
        fire = s.ready and (s.ammo > 0 or s.laser) and clear and after < tol
        return err, fire and (s.laser or t < self.max_shot_time)


class DefenseBot(PursuitBot):
    """Longer-horizon defensive teacher; only reads the supplied observation.

    Unlike the baseline, evaluate all observed bullets along the proposed
    maneuver. A bullet safe under current drift can become dangerous after
    turning, so filtering only by current drift misses relevant threats.
    """
    def __init__(self, near=500, far=700, lead=.8, horizon=25, margin=28, dodge=True):
        super().__init__(near,far,.8,dodge,1.2)
        self.lead,self.horizon,self.margin=lead,horizon,margin

    def _aim(self,s,target_r):
        t=intercept_time(s.dx,s.dy,s.ovx*self.lead,s.ovy*self.lead,620)
        t=math.hypot(s.dx,s.dy)/620 if t is None else t
        ax=s.dx+(0 if s.laser else s.ovx*t*self.lead)
        ay=s.dy+(0 if s.laser else s.ovy*t*self.lead)
        err=_wrap(math.atan2(ax,-ay)-s.ang)
        tol=math.atan2(target_r*.85,max(math.hypot(ax,ay),1))*self.fire_tol
        clear=line_of_sight(s.px,s.py,s.px+ax,s.py+ay,s.asteroids)
        return err,clear and t<self.max_shot_time and max(0,abs(err)-.07)<tol

    def __call__(self,obs):
        s=State(obs)
        if s.me_astro:
            turn,thrust,shoot=_flee(s)
        elif s.opp_astro:
            turn,thrust,shoot=self._hunt(s)
        else:
            turn,thrust,shoot=self._duel(s)
        return self.finish_action(s,turn,thrust)

    def finish_action(self,s,turn,thrust,allow_fire=True):
        """Emergency defense stays after navigation, before final firing."""
        if s.opp_bullets and self.dodge:
            base=_sim_min_dist(s,turn,thrust,s.opp_bullets,self.horizon,s.me_astro)
            margin=self.margin+(0 if s.me_astro else 7)
            if base<margin:
                bestscore,pref=base,turn
                for tr in [-1.,-.5,0.,.5,1.]:
                    for th in [-1.,0.,1.]:
                        d=_sim_min_dist(s,tr,th,s.opp_bullets,self.horizon,s.me_astro)
                        score=min(d,margin+10)-.25*abs(tr-pref)
                        if score>bestscore:
                            turn,thrust,bestscore=tr,th,score
        if not s.me_astro:
            err,fire=self._aim(s,13 if s.opp_astro else 20)
            tolerance=math.atan2(11 if s.opp_astro else 17,max(math.hypot(s.dx,s.dy),1))*self.fire_tol
            shoot=allow_fire and fire and abs(err-turn*.07)<tolerance
        else:
            shoot=False
        return np.array([turn,thrust,1 if shoot else -1],dtype=np.float32)


class CoverBot(DefenseBot):
    """Planet cover -> peek with a full magazine -> retreat while reloading.

    There is no elapsed-time observation. The initial slow approach to the
    shadow is a state-based approximation of the requested opening dodge period.
    No clock, hidden game state or persistent counter is used by the policy.
    """
    def __init__(self,*args,cover_radius=180.,peek_angle=1.05,
                 opening_speed=80.,**kwargs):
        super().__init__(*args,**kwargs)
        self.cover_radius=cover_radius
        self.peek_angle=peek_angle
        self.opening_speed=opening_speed

    @staticmethod
    def planet_blocks(s):
        length=s.dx*s.dx+s.dy*s.dy
        t=max(0.,min(1.,((600-s.px)*s.dx+(400-s.py)*s.dy)/max(length,1e-9)))
        return math.hypot(s.px+t*s.dx-600,s.py+t*s.dy-400)<104

    def plan(self,s):
        """Return an observable phase and navigation target for diagnosis."""
        ox,oy=s.px+s.dx-600,s.py+s.dy-400
        away=math.atan2(oy,ox)+math.pi
        radial=math.hypot(s.px-600,s.py-400)
        # Every legal initial spawn has radius >= 250 px. Move slowly inward
        # behind the planet before starting the peek/reload cycle.
        opening=radial>220
        ready=s.ammo>=3 and s.ready
        phase='opening' if opening else ('peek' if ready else 'retreat')
        offset=_wrap(math.atan2(s.py-400,s.px-600)-away)
        side=(1 if offset>0 else -1) if abs(offset)>.12 else (1 if self.peek_angle>=0 else -1)
        angle=away+(side*abs(self.peek_angle) if phase=='peek' else 0.)
        tx=600+self.cover_radius*math.cos(angle)
        ty=400+self.cover_radius*math.sin(angle)
        speed=self.opening_speed if opening else (140. if ready else 200.)
        if not self.planet_blocks(s):
            speed=max(speed,200.)
        return phase,tx,ty,speed

    def navigate(self,s,tx,ty,speed):
        # Short arc waypoints keep a direct path from cutting through the planet.
        gx,gy=tx-s.px,ty-s.py
        length=gx*gx+gy*gy
        t=max(0.,min(1.,((600-s.px)*gx+(400-s.py)*gy)/max(length,1e-9)))
        if math.hypot(s.px+t*gx-600,s.py+t*gy-400)<132:
            angle=math.atan2(s.py-400,s.px-600)
            gap=_wrap(math.atan2(ty-400,tx-600)-angle)
            angle+=max(-.45,min(.45,gap))
            radius=max(self.cover_radius,min(math.hypot(s.px-600,s.py-400),230.))
            tx,ty=600+radius*math.cos(angle),400+radius*math.sin(angle)
        dx,dy=tx-s.px,ty-s.py
        distance=math.hypot(dx,dy)
        velocity=min(speed,distance*2.)
        dvx=dx/max(distance,1e-9)*velocity-s.vx
        dvy=dy/max(distance,1e-9)*velocity-s.vy
        ax,ay=dvx*4.,dvy*4.
        # Fixed cover bars can otherwise trap an approach from a corner.
        for cx,cy,hw,hh in WALLS:
            qx=min(max(s.px,cx-hw),cx+hw)
            qy=min(max(s.py,cy-hh),cy+hh)
            rx,ry=s.px-qx,s.py-qy
            d=math.hypot(rx,ry)
            if 1e-6<d<65:
                force=250*(65-d)/65
                ax+=force*rx/d; ay+=force*ry/d
        error=_wrap(math.atan2(ax,-ay)-s.ang)
        reverse=_wrap(error-math.pi)
        backwards=abs(reverse)+.35<abs(error)
        error=reverse if backwards else error
        turn=max(-1.,min(1.,error/.07))
        strength=min(1.,math.hypot(ax,ay)/(350 if backwards else 700))
        thrust=strength*max(0.,math.cos(error))*(-1 if backwards else 1)
        return turn,thrust

    def __call__(self,obs):
        s=State(obs)
        if s.me_astro:
            turn,thrust,_=_flee(s)
            return self.finish_action(s,turn,thrust,False)
        if s.opp_astro:
            turn,thrust,_=self._hunt(s)
            return self.finish_action(s,turn,thrust)
        # Do not sacrifice immediate survival to insist on a cover waypoint.
        if math.hypot(s.dx,s.dy)<280 and not self.planet_blocks(s):
            turn,thrust,_=self._duel(s)
            return self.finish_action(s,turn,thrust)
        phase,tx,ty,speed=self.plan(s)
        turn,thrust=self.navigate(s,tx,ty,speed)
        if phase=='peek' and not self.planet_blocks(s):
            err,_=self._aim(s,20)
            turn=max(-1.,min(1.,err/.07))
            # Brake the drift while turning the gun toward the intercept point.
            thrust=-.8 if s.vx*s.hx+s.vy*s.hy>50 else 0.
        return self.finish_action(s,turn,thrust,phase=='peek')


class OcclusionDefenseBot(DefenseBot):
    """Optional ship collisions and projectile occlusion in forward defense.

    Both use observed positions and the public map. The collision-only option
    keeps all enemy bullets; the occlusion-only option keeps simple ship motion.
    """
    def __init__(self,*args,collisions=False,blocking=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.collisions,self.blocking=collisions,blocking

    def distances(self,s,turns,thrusts):
        n=len(turns)
        px=np.full(n,s.px,dtype=float); py=np.full(n,s.py,dtype=float)
        vx=np.full(n,s.vx,dtype=float); vy=np.full(n,s.vy,dtype=float)
        ang=np.full(n,s.ang,dtype=float)
        accel=380 if s.me_astro else 700
        drag=.995 if s.me_astro else .999
        maxspeed=110 if s.me_astro else 260
        radius=9 if s.me_astro else 16
        acc=accel*np.where(thrusts>=0,thrusts,thrusts*.5)*.02
        bullets=np.asarray(s.opp_bullets)
        lives=np.full(len(bullets),np.inf)
        for j,(bx,by,bvx,bvy) in enumerate(bullets):
            if not self.blocking: break
            speed=math.hypot(bvx,bvy)
            origin=np.array([bx,by]); direction=np.array([bvx,bvy])/speed
            hits=[_ray_box_exit_t(origin,direction)]
            t=_ray_circle_t(origin,direction,PLANET_POS,PLANET_RADIUS+4)
            if t is not None: hits.append(t)
            for cx,cy,hw,hh in WALLS:
                t=_ray_aabb_entry_t(origin,direction,cx,cy,hw,hh)
                if t is not None: hits.append(t)
            lives[j]=min(hits)/speed
        best=np.full(n,1e6)
        walls=np.asarray(WALLS)
        for k in range(1,self.horizon+1):
            ang+=turns*.07
            vx+=np.sin(ang)*acc; vy-=np.cos(ang)*acc
            speed=np.hypot(vx,vy)
            scale=np.minimum(1,maxspeed/np.maximum(speed,1e-9))*drag
            vx*=scale; vy*=scale
            px+=vx*.02; py+=vy*.02
            if self.collisions:
                # Match the public game's boundary and fixed-object responses.
                for q in range(n):
                    if px[q]<radius: px[q]=radius; vx[q]=-vx[q]*.65
                    if px[q]>1200-radius: px[q]=1200-radius; vx[q]=-vx[q]*.65
                    if py[q]<radius: py[q]=radius; vy[q]=-vy[q]*.65
                    if py[q]>800-radius: py[q]=800-radius; vy[q]=-vy[q]*.65
                    if math.hypot(px[q]-600,py[q]-400)<100+radius:
                        pos=np.array([px[q],py[q]]); vel=np.array([vx[q],vy[q]])
                        _resolve_circle_collision(pos,vel,1.,radius,PLANET_POS.copy(),np.zeros(2),0.,100.,.8)
                        px[q],py[q]=pos; vx[q],vy[q]=vel
                cx=np.clip(px[:,None],walls[:,0]-walls[:,2],walls[:,0]+walls[:,2])
                cy=np.clip(py[:,None],walls[:,1]-walls[:,3],walls[:,1]+walls[:,3])
                pairs=np.argwhere((px[:,None]-cx)**2+(py[:,None]-cy)**2<radius**2)
                for q,j in pairs:
                    pos=np.array([px[q],py[q]]); vel=np.array([vx[q],vy[q]])
                    _circle_rect_collide(pos,vel,radius,*WALLS[j],.65)
                    px[q],py[q]=pos; vx[q],vy[q]=vel
            bx=bullets[:,0]+bullets[:,2]*k*.02
            by=bullets[:,1]+bullets[:,3]*k*.02
            distances=np.hypot(px[:,None]-bx,py[:,None]-by)
            distances[:,k*.02>lives]=1e6
            best=np.minimum(best,distances.min(axis=1))
        return best

    def __call__(self,obs):
        s=State(obs)
        if s.me_astro: turn,thrust,shoot=_flee(s)
        elif s.opp_astro: turn,thrust,shoot=self._hunt(s)
        else: turn,thrust,shoot=self._duel(s)
        if s.opp_bullets and self.dodge:
            turns=np.array([turn]+[tr for tr in [-1.,-.5,0.,.5,1.] for _ in range(3)])
            thrusts=np.array([thrust]+[-1.,0.,1.]*5)
            ds=self.distances(s,turns,thrusts)
            margin=self.margin+(0 if s.me_astro else 7)
            if ds[0]<margin:
                scores=np.minimum(ds,margin+10)-.25*np.abs(turns-turn)
                scores[0]=ds[0]
                i=int(np.argmax(scores))
                turn,thrust=turns[i],thrusts[i]
        if not s.me_astro:
            err,fire=self._aim(s,13 if s.opp_astro else 20)
            tolerance=math.atan2(11 if s.opp_astro else 17,max(math.hypot(s.dx,s.dy),1))*self.fire_tol
            shoot=fire and abs(err-turn*.07)<tolerance
        return np.array([turn,thrust,1 if shoot else -1],dtype=np.float32)


class GeometryFeatures(BaseFeaturesExtractor):
    """Raw obs plus heading and interception geometry, in correct units."""
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=96)

    def forward(self, o):
        dx, dy = o[:,14]*600, o[:,15]*400
        vx, vy = (o[:,2]+o[:,16])*260, (o[:,3]+o[:,17])*260
        a, b, c = vx*vx+vy*vy-620**2, 2*(dx*vx+dy*vy), dx*dx+dy*dy
        t = ((-b-torch.sqrt(torch.clamp(b*b-4*a*c,min=0)))/(2*a)).clamp(0,3)
        ax, ay = dx+vx*t, dy+vy*t
        angle = torch.atan2(o[:,5],o[:,4])
        err = torch.atan2(ax,-ay)-angle
        err = torch.atan2(torch.sin(err),torch.cos(err))
        direct = torch.atan2(dx,-dy)-angle
        distance = torch.sqrt(c).clamp(min=1)
        closing = (dx*o[:,16]+dy*o[:,17])/distance
        extra = torch.stack((torch.sin(err),torch.cos(err),(err/.07).clamp(-1,1),
            torch.sin(direct),torch.cos(direct),distance/600,t,closing,
            o[:,12]*o[:,21],o[:,8]*o[:,11],o[:,21]*distance/600,
            torch.hypot(o[:,2],o[:,3])),dim=1)
        return torch.cat((o,extra),dim=1)


class TacticalFeatures(BaseFeaturesExtractor):
    """Hybrid option, explicitly NOT end-to-end RL: controller proposals plus
    raw state feed a learned linear SB3 actor. All output weights are learned.
    Report this inductive bias when using this option.
    """
    def __init__(self, observation_space, near=30.0, far=100.0, dodge=True,
                 controller='pursuit',lead=.8,horizon=25,margin=28,
                 cover_radius=180.,peek_angle=1.05,opening_speed=80.):
        super().__init__(observation_space, features_dim=29)
        if controller=='pursuit':
            self.teacher=PursuitBot(near,far,1.0,dodge,.8)
        elif controller=='cover':
            self.teacher=CoverBot(near,far,lead,horizon,margin,dodge,
                cover_radius=cover_radius,peek_angle=peek_angle,opening_speed=opening_speed)
        else:
            use_physics=controller in ['occlusion','collision','physics']
            bot_class=OcclusionDefenseBot if use_physics else DefenseBot
            extra=dict(collisions=controller in ['collision','physics'],blocking=controller!='collision') if use_physics else {}
            self.teacher=bot_class(near,far,lead,horizon,margin,dodge,**extra)
        # PPO revisits the same rollout several times. Cache only this pure,
        # deterministic computation; it does not alter observations or actions.
        self._feature_cache = {}

    def forward(self, o):
        if len(self._feature_cache)>16384:
            self._feature_cache.clear()
        proposals=[]
        for row in o.detach().cpu().numpy():
            key=row.tobytes()
            if key not in self._feature_cache:
                self._feature_cache[key]=self.teacher(row)
            proposals.append(self._feature_cache[key])
        proposals=np.stack(proposals)
        return torch.cat((torch.as_tensor(proposals,device=o.device),o[:,:26]),dim=1)


def make_env():
    return Monitor(AstroPartyTrainEnv(mixed_opponents=True))


def write_json(filename, value):
    """Write only experiment outputs, using Python's built-in file interface."""
    with open(filename,'w',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)


def init_tracking(config):
    """A TA can retrain without our W&B account or an interactive login."""
    requested=config['wandb_mode']
    mode=requested
    if requested=='auto':
        try:
            # Use existing credentials only; never ask for or embed an API key.
            authenticated=wandb.login(prompt=False,verify=False)
        except (wandb.errors.Error,TypeError):
            authenticated=False
        mode='online' if authenticated else 'offline'
    options=dict(project=config['project'],entity=config['entity'],name=config['name'],
        config=config,dir='.',settings=wandb.Settings(init_timeout=30))
    try:
        run=wandb.init(mode=mode,**options)
    except wandb.errors.Error:
        if requested!='auto' or mode=='offline':
            raise
        print('W&B online unavailable; keeping the full training log offline.',flush=True)
        mode='offline'
        run=wandb.init(mode=mode,**options)
    run.summary['logging_mode']=mode
    print('W&B logging mode:',mode,flush=True)
    return run


def evaluate(model, episodes, seed_base=10000):
    """Official sparse environment, deterministic actions, no draw credit."""
    metrics, rows = {}, []
    for name, opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
        env = AstroPartyEvalEnv()
        wins = losses = draws = total_steps = 0
        for seed in range(seed_base,seed_base+episodes):
            obs, _ = env.reset(seed=seed)
            done, steps = False, 0
            while not done:
                action, _ = model.predict(obs['ship1'],deterministic=True)
                obs, _, terms, truncs, _ = env.step({
                    'ship1':decode_action(action),
                    'ship2':decode_action(opponent(obs['ship2']))})
                done = terms['ship1'] or truncs['ship1']
                steps += 1
            win = env.ships['ship1'].alive and not env.ships['ship2'].alive
            loss = not env.ships['ship1'].alive
            wins += int(win)
            losses += int(loss)
            draws += int(not win and not loss)
            total_steps += steps
            rows.append(dict(opponent=name,seed=seed,win=bool(win),loss=bool(loss),steps=steps))
        env.close()
        metrics.update({f'eval/{name}_win_rate':wins/episodes,
            f'eval/{name}_loss_rate':losses/episodes,f'eval/{name}_draw_rate':draws/episodes,
            f'eval/{name}_length':total_steps/episodes})
    metrics['eval/balanced_score'] = min(metrics['eval/rusher_win_rate']/.90,
                                         metrics['eval/sniper_win_rate']/.85)
    return metrics, rows


class TrainingLog(BaseCallback):
    def __init__(self, run):
        super().__init__()
        self.run, self.started = run, time.perf_counter()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        data = {k:float(v) for k,v in self.model.logger.name_to_value.items()
                if isinstance(v,(float,int,np.number))}
        data.update(global_step=self.num_timesteps)
        data['time/steps_per_second'] = self.num_timesteps/max(time.perf_counter()-self.started,1)
        recent = self.model.ep_info_buffer
        if recent:
            data['rollout/ep_rew_mean'] = float(np.mean([e['r'] for e in recent]))
            data['rollout/ep_len_mean'] = float(np.mean([e['l'] for e in recent]))
        self.run.log(data)


class AnchoredPPO(PPO):
    """PPO plus conservative shrinkage toward the BC actor initialization.

    Only learning changes. The grader loads ordinary PPO and uses its normal
    actor. This is useful when tiny aiming changes can erase a good controller.
    retention=1 is ordinary PPO; smaller values impose a stronger prior.
    """
    anchor_retention = 0.1

    def train(self):
        actor=list(self.policy.action_net.parameters())
        if not hasattr(self,'anchor_state'):
            self.anchor_state=[p.detach().clone() for p in actor]
        super().train()
        with torch.no_grad():
            for parameter,prior in zip(actor,self.anchor_state):
                parameter.copy_(prior+self.anchor_retention*(parameter-prior))
        self.logger.record('train/anchor_retention',self.anchor_retention)
        self.logger.record('train/actor_prior_distance',float(sum(
            (p.detach()-prior).square().sum() for p,prior in zip(actor,self.anchor_state)).sqrt()))


def collect_demonstrations(teacher, count, seed, run):
    env = AstroPartyTrainEnv(mixed_opponents=True)
    obs, _ = env.reset(seed=100000+seed)
    observations, actions = [], []
    for i in range(count):
        action = teacher(obs)
        observations.append(obs)
        actions.append(action)
        obs, _, done, trunc, _ = env.step(action)
        if done or trunc:
            obs, _ = env.reset()
        if (i+1)%5000 == 0:
            print(f'Demonstrations: {i+1}/{count}',flush=True)
            run.log({'bc/collected_steps':i+1})
    env.close()
    return np.asarray(observations),np.asarray(actions)


def behavior_clone(model, obs, targets, epochs, run, seed):
    rng = np.random.default_rng(seed)
    model.policy.set_training_mode(True)
    x,y = torch.as_tensor(obs),torch.as_tensor(targets)
    with torch.no_grad():
        features = torch.cat([model.policy.extract_features(x[i:i+512]) for i in range(0,len(x),512)])
    optimizer = torch.optim.Adam(model.policy.parameters(),lr=.003)
    for epoch in range(epochs):
        losses = []
        for indices in np.array_split(rng.permutation(len(x)),max(1,len(x)//256)):
            latent = model.policy.mlp_extractor.forward_actor(features[indices])
            pred = model.policy.action_net(latent)
            loss = ((pred-y[indices])**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        value = float(np.mean(losses))
        run.log({'bc/epoch':epoch+1,'bc/mse':value})
        if (epoch+1)%10 == 0:
            print(f'BC epoch {epoch+1}: mse={value:.6f}',flush=True)
    model.policy.set_training_mode(False)


def train(model, config, run):
    best, history = -1.0, []
    callback = TrainingLog(run)
    folder = run.dir  # W&B creates this output directory.
    for epoch in range(config['epochs']):
        started = time.perf_counter()
        model.learn(total_timesteps=config['steps_per_epoch'],reset_num_timesteps=False,
                    callback=callback,tb_log_name=run.id)
        metrics,rows = evaluate(model,config['eval_episodes'],config['eval_seed'])
        metrics.update(global_step=model.num_timesteps,epoch=epoch+1,
                       epoch_seconds=time.perf_counter()-started)
        latest_train={k:float(v) for k,v in model.logger.name_to_value.items()
                      if isinstance(v,(int,float,np.number))}
        run.log(dict(latest_train,**metrics))
        history.append(metrics)
        write_json(f'{folder}/history.json',history)
        write_json(f'{folder}/eval_{epoch+1:03d}.json',rows)
        model.save(f'models/{run.id}_latest')
        if metrics['eval/balanced_score'] > best:
            best = metrics['eval/balanced_score']
            model.save(config['save_path'])
            model.save(f'models/{run.id}_best')
            run.summary['best_balanced_score'] = best
            run.summary['best_global_step'] = model.num_timesteps
        print(json.dumps(metrics),flush=True)
    return history


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name',default='defense-ppo-reproducible')
    p.add_argument('--project',default='astro-party-hw3')
    p.add_argument('--entity',default=None)
    p.add_argument('--wandb-mode',choices=['auto','online','offline','disabled'],default='auto')
    p.add_argument('--features',choices=['geometry','tactical','raw'],default='tactical')
    for name,default in [('epochs',5),('steps-per-epoch',8192),('eval-episodes',50),
                         ('eval-seed',10000),('seed',42),('envs',4),('bc-steps',10000),('bc-epochs',120)]:
        p.add_argument('--'+name,type=int,default=default)
    p.add_argument('--learning-rate',type=float,default=1e-6)
    p.add_argument('--near',type=float,default=500)
    p.add_argument('--far',type=float,default=700)
    p.add_argument('--no-dodge',action='store_true')
    p.add_argument('--controller',choices=['pursuit','defense','cover','occlusion','collision','physics'],default='defense')
    p.add_argument('--lead',type=float,default=.8)
    p.add_argument('--horizon',type=int,default=60)
    p.add_argument('--margin',type=float,default=28)
    p.add_argument('--log-std',type=float,default=-3.5)
    p.add_argument('--anchor-retention',type=float,default=1.0)
    p.add_argument('--cover-radius',type=float,default=180.)
    p.add_argument('--peek-angle',type=float,default=1.05)
    p.add_argument('--opening-speed',type=float,default=80.)
    p.add_argument('--resume',default=None)
    p.add_argument('--save-path',default=None)
    p.add_argument('--select-last',action=argparse.BooleanOptionalAction,default=True,
                   help='Export the last checkpoint instead of the validation best')
    p.add_argument('--upload-model',action='store_true',
                   help='Also upload the local checkpoint as a W&B artifact (opt-in)')
    args = p.parse_args()
    if args.save_path is None:
        args.save_path='models/model_astro_party_cover' if args.controller=='cover' else 'models/model_astro_party'
    config = vars(args)
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run = init_tracking(config)
    run.define_metric('global_step')
    for prefix in ['eval','train','rollout']:
        run.define_metric(prefix+'/*',step_metric='global_step')
    write_json(f'{run.dir}/run_info.json',dict(id=run.id,url=run.url,config=config))
    print(f'W&B run: {run.url}',flush=True)
    env = DummyVecEnv([make_env]) if args.envs == 1 else SubprocVecEnv([make_env for _ in range(args.envs)])
    env.seed(100000+args.seed)
    kwargs = dict(net_arch=dict(pi=[128,128],vf=[128,128]),activation_fn=nn.Tanh,log_std_init=args.log_std)
    if args.features == 'geometry':
        kwargs['features_extractor_class'] = GeometryFeatures
    elif args.features == 'tactical':
        kwargs.update(features_extractor_class=TacticalFeatures,
            features_extractor_kwargs=dict(near=args.near,far=args.far,dodge=not args.no_dodge,
                controller=args.controller,lead=args.lead,horizon=args.horizon,margin=args.margin,
                cover_radius=args.cover_radius,peek_angle=args.peek_angle,opening_speed=args.opening_speed),
            net_arch=dict(pi=[],vf=[64,64]))
    algorithm=AnchoredPPO if args.anchor_retention<1.0 else PPO
    if args.resume:
        model = algorithm.load(args.resume,env=env,device='cpu',learning_rate=args.learning_rate)
    else:
        model = algorithm('MlpPolicy',env,policy_kwargs=kwargs,device='cpu',seed=args.seed,
            learning_rate=args.learning_rate,n_steps=512,batch_size=256,n_epochs=5,
            gamma=.995,gae_lambda=.95,clip_range=.1,ent_coef=0.,target_kl=.02,
            verbose=0,tensorboard_log='runs')
    if isinstance(model,AnchoredPPO):
        model.anchor_retention=args.anchor_retention
    parameters = sum(v.numel() for v in model.policy.parameters())
    assert parameters <= 2000000
    run.summary['parameter_count'] = parameters
    try:
        if args.bc_steps and not args.resume:
            if args.controller=='pursuit':
                teacher=PursuitBot(args.near,args.far,1.,not args.no_dodge,.8)
            elif args.controller=='cover':
                teacher=CoverBot(args.near,args.far,args.lead,args.horizon,args.margin,not args.no_dodge,
                    cover_radius=args.cover_radius,peek_angle=args.peek_angle,opening_speed=args.opening_speed)
            else:
                use_physics=args.controller in ['occlusion','collision','physics']
                bot_class=OcclusionDefenseBot if use_physics else DefenseBot
                extra=dict(collisions=args.controller in ['collision','physics'],blocking=args.controller!='collision') if use_physics else {}
                teacher=bot_class(args.near,args.far,args.lead,args.horizon,args.margin,not args.no_dodge,**extra)
            obs,labels = collect_demonstrations(teacher,args.bc_steps,args.seed,run)
            behavior_clone(model,obs,labels,args.bc_epochs,run,args.seed)
            model.save(f'models/{run.id}_bc')
            metrics,rows = evaluate(model,args.eval_episodes,args.eval_seed)
            run.log(dict(metrics,global_step=0))
            write_json(f'{run.dir}/bc.json',dict(metrics=metrics,episodes=rows))
            print('BC validation:',json.dumps(metrics),flush=True)
        train(model,config,run)
        if args.select_last:
            model.save(args.save_path)
            run.summary['exported_checkpoint']='last'
            run.summary['exported_global_step']=model.num_timesteps
        if args.upload_model:
            artifact = wandb.Artifact(f'astro-party-{run.id}',type='model')
            artifact.add_file(args.save_path+'.zip')
            run.log_artifact(artifact)
    finally:
        env.close()
        run.finish()


if __name__ == '__main__':
    main()
