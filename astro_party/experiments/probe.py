"""Diagnostic only: compare interpretable teachers, never a submitted policy."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from envs.eval_astro_party_env import AstroPartyEvalEnv
from envs.game import decode_action, SHIP_RADIUS, ASTRONAUT_RADIUS, BULLET_RADIUS, BULLET_SPEED
from models.sniper_bot import RuleBot, State, _wrap, TURN_STEP, line_of_sight, intercept_time
from models.sniper_bot import sniper_bot_action
from models.rusher_bot import rusher_bot_action


class PursuitBot(RuleBot):
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


def play(bot, opponent, seed):
    env = AstroPartyEvalEnv()
    obs, _ = env.reset(seed=seed)
    steps = 0
    while True:
        obs, _, term, trunc, _ = env.step({
            'ship1': decode_action(bot(obs['ship1'])),
            'ship2': decode_action(opponent(obs['ship2']))})
        steps += 1
        if term['ship1'] or trunc['ship1']:
            break
    win = env.ships['ship1'].alive and not env.ships['ship2'].alive
    loss = not env.ships['ship1'].alive
    env.close()
    return int(win), int(loss), steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=12)
    p.add_argument('--seed-base', type=int, default=30000)
    p.add_argument('--variant', default='all')
    a = p.parse_args()
    candidates = {
        'sniper': sniper_bot_action,
        'rush_dodge': RuleBot(80, 160, 1.2, True, .8),
        'pursuit': PursuitBot(30, 100, 1.0, True, .8),
        'pursuit_nododge': PursuitBot(30, 100, 1.0, False, .8),
        'pursuit_mid': PursuitBot(140, 260, 1.0, True, .8),
    }
    results = {}
    for name, bot in candidates.items():
        if a.variant != 'all' and name != a.variant:
            continue
        results[name] = {}
        for opponent_name, opponent in [('rusher', rusher_bot_action), ('sniper', sniper_bot_action)]:
            start = time.perf_counter()
            episodes = [play(bot, opponent, s) for s in range(a.seed_base, a.seed_base+a.episodes)]
            result = dict(wins=sum(x[0] for x in episodes), losses=sum(x[1] for x in episodes),
                          episodes=episodes, seconds=time.perf_counter()-start)
            results[name][opponent_name] = result
            print(name, opponent_name, result['wins'], result['losses'], round(result['seconds'], 1), flush=True)
            Path('results/probe_'+a.variant+'.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
