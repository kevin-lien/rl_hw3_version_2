"""Obstacle-aware forward defense. All inputs come from the public observation."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from experiments.defense import DefenseBot
from experiments.probe import play
from models.sniper_bot import State,_flee,sniper_bot_action
from models.rusher_bot import rusher_bot_action
from envs.game import (WALLS,PLANET_POS,PLANET_RADIUS,_ray_circle_t,
    _ray_aabb_entry_t,_ray_box_exit_t,_circle_rect_collide,
    _resolve_circle_collision)


class PhysicsBot(DefenseBot):
    def __init__(self,collisions=True,blocking=True,**kwargs):
        super().__init__(**kwargs)
        self.collisions,self.blocking=collisions,blocking

    def distances(self,s,turns,thrusts):
        n=len(turns)
        px=np.full(n,s.px,dtype=float); py=np.full(n,s.py,dtype=float)
        vx=np.full(n,s.vx,dtype=float); vy=np.full(n,s.vy,dtype=float)
        ang=np.full(n,s.ang,dtype=float)
        radius=9 if s.me_astro else 16
        accel=380 if s.me_astro else 700
        drag=.995 if s.me_astro else .999
        maxspeed=110 if s.me_astro else 260
        acc=accel*np.where(thrusts>=0,thrusts,thrusts*.5)*.02
        bullets=np.asarray(s.opp_bullets)
        lives=np.full(len(bullets),np.inf)
        if self.blocking:
            for j,(bx,by,bvx,bvy) in enumerate(bullets):
                speed=math.hypot(bvx,bvy)
                origin=np.array([bx,by]); direction=np.array([bvx,bvy])/speed
                hits=[_ray_box_exit_t(origin,direction)]
                t=_ray_circle_t(origin,direction,PLANET_POS,PLANET_RADIUS+4)
                if t is not None: hits.append(t)
                for cx,cy,hw,hh in WALLS:
                    t=_ray_aabb_entry_t(origin,direction,cx,cy,hw,hh)
                    if t is not None: hits.append(t)
                lives[j]=min(hits)/speed
        walls=np.asarray(WALLS)
        best=np.full(n,1e6)
        for k in range(1,self.horizon+1):
            ang+=turns*.07
            vx+=np.sin(ang)*acc; vy-=np.cos(ang)*acc
            speed=np.hypot(vx,vy)
            scale=np.minimum(1,maxspeed/np.maximum(speed,1e-9))*drag
            vx*=scale; vy*=scale
            px+=vx*.02; py+=vy*.02
            if self.collisions:
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
        if s.opp_bullets:
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
            shoot=fire and abs(err-turn*.07)<math.atan2(11 if s.opp_astro else 17,max(math.hypot(s.dx,s.dy),1))*self.fire_tol
        return np.array([turn,thrust,1 if shoot else -1],dtype=np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=30)
    p.add_argument('--seed-base',type=int,default=10000)
    p.add_argument('--horizon',type=int,default=40)
    a=p.parse_args()
    results=[]
    for collisions,blocking in [(False,True),(True,False),(True,True)]:
        bot=PhysicsBot(collisions=collisions,blocking=blocking,horizon=a.horizon,margin=28,lead=.8,near=500,far=700)
        row=dict(collisions=collisions,blocking=blocking,horizon=a.horizon)
        for name,opponent in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
            games=[play(bot,opponent,s) for s in range(a.seed_base,a.seed_base+a.episodes)]
            row[name]=sum(x[0] for x in games)/len(games)
            print(row,flush=True)
        results.append(row)
        Path(f'results/physics_h{a.horizon}.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__': main()
