"""Physics-aware teacher experiments; diagnostic seeds only."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from experiments.probe import play
from models.sniper_bot import (RuleBot,State,_wrap,line_of_sight,_flee,
    _threatening,_sim_min_dist,TURN_STEP,intercept_time)
from models.sniper_bot import sniper_bot_action
from models.rusher_bot import rusher_bot_action
from envs.game import BULLET_SPEED,SHIP_RADIUS,ASTRONAUT_RADIUS,BULLET_RADIUS


class AdvancedBot(RuleBot):
    def __init__(self, lead=1., horizon=30, near=380,far=620,tol=.8,**kwargs):
        super().__init__(near,far,tol,True,1.2)
        self.lead,self.horizon=lead,horizon

    def _aim(self,s,target_r):
        t=intercept_time(s.dx,s.dy,s.ovx*self.lead,s.ovy*self.lead,620)
        t=math.hypot(s.dx,s.dy)/620 if t is None else t
        ax=s.dx+(0 if s.laser else s.ovx*t*self.lead)
        ay=s.dy+(0 if s.laser else s.ovy*t*self.lead)
        err=_wrap(math.atan2(ax,-ay)-s.ang)
        tol=math.atan2(target_r*.85,max(math.hypot(ax,ay),1))*self.fire_tol
        clear=line_of_sight(s.px,s.py,s.px+ax,s.py+ay,s.asteroids)
        return err, (clear and t<self.max_shot_time and max(0,abs(err)-.07)<tol)

    def __call__(self,obs):
        s=State(obs)
        if s.me_astro:
            turn,thrust,shoot=_flee(s)
        elif s.opp_astro:
            turn,thrust,shoot=self._hunt(s)
        else:
            turn,thrust,shoot=self._duel(s)
        radius=(ASTRONAUT_RADIUS if s.me_astro else SHIP_RADIUS)+BULLET_RADIUS
        threats=_threatening(s,s.opp_bullets,radius+15)
        if threats:
            margin=radius+15
            base=_sim_min_dist(s,turn,thrust,threats,self.horizon,s.me_astro)
            if base<margin:
                bestscore=base
                pref=turn
                for tr in [-1.,-.5,0.,.5,1.]:
                    for th in [-1.,0.,1.]:
                        d=_sim_min_dist(s,tr,th,threats,self.horizon,s.me_astro)
                        score=min(d,margin+5)-.5*abs(tr-pref)
                        if score>bestscore:
                            turn,thrust,bestscore=tr,th,score
        # Actual final action can change the heading used for a shot.
        if not s.me_astro:
            err,fire=self._aim(s,13 if s.opp_astro else 20)
            shoot=fire and abs(err-turn*.07)<math.atan2(11 if s.opp_astro else 17,max(math.hypot(s.dx,s.dy),1))*self.fire_tol
        return np.array([turn,thrust,1 if shoot else -1],dtype=np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed-base',type=int,default=32000)
    a=p.parse_args()
    results=[]
    for horizon in [18,30,45]:
        for lead in [.5,.8,1.,1.2]:
            bot=AdvancedBot(lead=lead,horizon=horizon)
            row=dict(lead=lead,horizon=horizon)
            for name,opp in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
                games=[play(bot,opp,s) for s in range(a.seed_base,a.seed_base+a.episodes)]
                row[name]=sum(g[0] for g in games)/len(games)
            print(row,flush=True)
            results.append(row)
            Path('results/advanced.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
