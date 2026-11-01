import argparse
import json
import math
from pathlib import Path
import numpy as np
from experiments.probe import play
from experiments.advanced import AdvancedBot
from models.sniper_bot import State,_flee,_sim_min_dist,sniper_bot_action
from models.rusher_bot import rusher_bot_action


class DefenseBot(AdvancedBot):
    def __init__(self,margin=35,**kwargs):
        super().__init__(**kwargs)
        self.margin=margin

    def __call__(self,obs):
        s=State(obs)
        if s.me_astro:
            turn,thrust,shoot=_flee(s)
        elif s.opp_astro:
            turn,thrust,shoot=self._hunt(s)
        else:
            turn,thrust,shoot=self._duel(s)
        bullets=s.opp_bullets
        if bullets:
            base=_sim_min_dist(s,turn,thrust,bullets,self.horizon,s.me_astro)
            margin=self.margin+(0 if s.me_astro else 7)
            if base<margin:
                bestscore=base
                pref=turn
                for tr in [-1.,-.5,0.,.5,1.]:
                    for th in [-1.,0.,1.]:
                        d=_sim_min_dist(s,tr,th,bullets,self.horizon,s.me_astro)
                        score=min(d,margin+10)-.25*abs(tr-pref)
                        if score>bestscore:
                            turn,thrust,bestscore=tr,th,score
        if not s.me_astro:
            err,fire=self._aim(s,13 if s.opp_astro else 20)
            shoot=fire and abs(err-turn*.07)<math.atan2(11 if s.opp_astro else 17,max(math.hypot(s.dx,s.dy),1))*self.fire_tol
        return np.array([turn,thrust,1 if shoot else -1],dtype=np.float32)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed-base',type=int,default=33000)
    a=p.parse_args()
    results=[]
    for horizon in [25,40,60]:
        for margin in [28,45,65]:
            bot=DefenseBot(horizon=horizon,margin=margin,lead=.8,near=500,far=700)
            row=dict(horizon=horizon,margin=margin)
            for name,opp in [('rusher',rusher_bot_action),('sniper',sniper_bot_action)]:
                games=[play(bot,opp,s) for s in range(a.seed_base,a.seed_base+a.episodes)]
                row[name]=sum(g[0] for g in games)/len(games)
            print(row,flush=True)
            results.append(row)
            Path('results/defense.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
