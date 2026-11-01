"""Training environment: a single-agent gym.Env where you fly ship1.

ship2 is flown by `opponent_fn`: any callable that takes ship2's observation
(Box(84,), same layout as yours, from ship2's point of view) and returns a
Box(3,) action. Use a rule-based bot (models/sniper_bot.py) or a trained
model wrapped as
`lambda obs: model.predict(obs, deterministic=True)[0]`, including a frozen copy
of your own model for self-play. Reassign `env.opponent_fn` whenever you like.

THE REWARD IS YOURS TO DESIGN: everything is in the weights and the `reward()`
method below. Grading does not use it; it uses envs/eval_astro_party_env.py's sparse
+1/-1 for the final kill.

The game itself (physics, weapons, observation layout) lives in envs/game.py.
Each game step returns a list of Events (kill, eject, asteroid_hit, pickup,
respawn). `reward()` turns them into a number. For other shaping, read the
game state directly: self.game.ships["ship1"].pos, .vel, .angle, .ammo,
.mode, self.game.bullets, self.game.asteroids, ... (see envs/game.py).

Action: Box(3,) [turn_rate, thrust, shoot_signal] in [-1, 1]; shoot_signal > 0 fires.
"""
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from envs.game import ARENA_W, FPS, MAX_SPEED, PLANET_POS, AstroPartyGame, decode_action, observation_space
from envs.render import Renderer

# ----------------------------------------------------------------- reward weights
KILL_REWARD = 20.0         # Final elimination dominates all auxiliary rewards.
EJECT_REWARD = 3.0
RESPAWN_REWARD = 0.0       # No reward for repeatedly losing and regaining a ship.
ASTEROID_HIT_REWARD = 0.0   # Avoid farming asteroids instead of winning.
PICKUP_REWARD = 0.1
APPROACH_COEF = 0.5
STEP_PENALTY = -0.002

# Opt-in fast-cover reward. At full speed the maximum 1500-step velocity
# bonus is 2.25, smaller than the 20-point final kill, and the per-step net
# reward stays negative. Existing checkpoints keep the original reward profile.
VELOCITY_REWARD_COEF = 0.0015
HUNT_APPROACH_COEF = 6.0  # Previous effective pilot-chase coefficient was 0.5 * 4 = 2.
FAST_COVER_APPROACH_COEF = 0.1


class AstroPartyTrainEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, opponent_fn=None, render_mode=None, random_spawn=True,
                 mixed_opponents=False, reward_profile='original',
                 velocity_coef=VELOCITY_REWARD_COEF,hunt_approach_coef=HUNT_APPROACH_COEF):
        # random_spawn=False starts every game from the same fixed spawn points, an
        # easier setting to begin with. Grading always uses random spawns.
        self.game = AstroPartyGame(random_spawn=random_spawn)
        self.renderer = Renderer(render_mode)
        self.render_mode = render_mode
        self.observation_space = observation_space()
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.opponent_fn = opponent_fn if opponent_fn is not None else (lambda obs: np.zeros(3, dtype=np.float32))
        self.mixed_opponents = mixed_opponents
        self.opponent_name = 'custom'
        if reward_profile not in ['original','fast_cover']:
            raise ValueError('Unknown reward profile: '+reward_profile)
        if not 0<=velocity_coef<-STEP_PENALTY:
            raise ValueError('velocity_coef must be nonnegative and smaller than the step penalty')
        self.reward_profile=reward_profile
        self.velocity_coef=velocity_coef
        self.hunt_approach_coef=hunt_approach_coef
        self.last_reward_parts={}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.mixed_opponents:
            from models.rusher_bot import rusher_bot_action
            from models.sniper_bot import sniper_bot_action
            use_sniper = self.np_random.random() < 0.5
            self.opponent_fn = sniper_bot_action if use_sniper else rusher_bot_action
            self.opponent_name = 'sniper' if use_sniper else 'rusher'
        self.game.reset(seed)
        self.prev_dist = self.distance()
        if self.render_mode == "human":
            self.render()
        return self.game.observe("ship1"), {}

    def step(self, action):
        opponent_action = self.opponent_fn(self.game.observe("ship2"))
        events = self.game.step({"ship1": decode_action(action), "ship2": decode_action(opponent_action)})
        reward = self.reward(events)
        if self.render_mode == "human":
            self.render()
        info = {'opponent': self.opponent_name,**self.last_reward_parts}
        if self.game.done or self.game.truncated:
            info['is_success'] = self.game.ships['ship1'].alive and not self.game.ships['ship2'].alive
            info['draw'] = not self.game.done
        return self.game.observe("ship1"), reward, self.game.done, self.game.truncated, info

    # ----------------------------------------------------------------- reward

    def reward(self, events) -> float:
        """Your reward for one step, given what happened in it (`events`)."""
        me = "ship1"
        r = STEP_PENALTY
        for e in events:
            if e.kind == "kill":
                r += KILL_REWARD if e.by == me else -KILL_REWARD
            elif e.kind == "eject":
                r += EJECT_REWARD if e.by == me else -EJECT_REWARD
            elif e.kind == "respawn" and e.by == me:
                r += RESPAWN_REWARD
            elif e.kind == "asteroid_hit" and e.by == me:
                r += ASTEROID_HIT_REWARD
            elif e.kind == "pickup" and e.by == me:
                r += PICKUP_REWARD

        # Shaping: get closer to the enemy.
        dist = self.distance()
        mine = self.game.ships['ship1'].mode == 'astronaut'
        theirs = self.game.ships['ship2'].mode == 'astronaut'
        speed=float(np.hypot(*self.game.ships['ship1'].vel))
        covered=self.planet_blocks()
        velocity_reward=0.0
        approach_coefficient=0.0
        if not self.game.done:
            # Closing helps a ship finish an ejected pilot; a pilot should flee.
            if self.reward_profile=='original':
                weight = -1.0 if mine else (4.0 if theirs else 1.0)
                approach_coefficient=APPROACH_COEF*weight
            else:
                approach_coefficient=(-APPROACH_COEF if mine else
                    self.hunt_approach_coef if theirs else FAST_COVER_APPROACH_COEF)
                if not mine:
                    ship,opponent=self.game.ships['ship1'],self.game.ships['ship2']
                    if theirs:
                        # Reward speed toward the pilot, not speeding past it.
                        direction=opponent.pos-ship.pos
                        distance=max(float(np.hypot(*direction)),1e-9)
                        effective_speed=max(0.,float(np.dot(ship.vel,direction))/distance)
                        context=1.0
                    else:
                        effective_speed=speed
                        context=1.0 if covered else .25
                    velocity_reward=self.velocity_coef*context*min(1.,effective_speed/MAX_SPEED)
                    r+=velocity_reward
            r += approach_coefficient * (self.prev_dist - dist)
        self.prev_dist = dist
        self.last_reward_parts=dict(speed=speed,velocity_reward=velocity_reward,
            covered=float(covered),hunt_mode=float(theirs and not mine),
            approach_coefficient=approach_coefficient,is_ship=float(not mine))
        return r

    def planet_blocks(self):
        me=self.game.ships['ship1'].pos
        delta=self.game.ships['ship2'].pos-me
        t=float(np.clip(np.dot(PLANET_POS-me,delta)/max(float(np.dot(delta,delta)),1e-9),0,1))
        return bool(np.hypot(*(me+t*delta-PLANET_POS))<104)

    def distance(self) -> float:
        """Distance between the two ships, in units of half the arena width."""
        p1, p2 = self.game.ships["ship1"].pos, self.game.ships["ship2"].pos
        return float(np.hypot(*(p2 - p1))) / (ARENA_W / 2)

    # ----------------------------------------------------------------- render

    def render(self):
        return self.renderer.draw(self.game)

    def close(self):
        self.renderer.close()
