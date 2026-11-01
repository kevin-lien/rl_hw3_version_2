"""Watch a checkpoint immediately using the original game renderer.

Example:
    python watch_astro_party.py --opponent sniper --seed 0
"""
import argparse

import torch

from eval_astro_party import load_model, play
from models.rusher_bot import rusher_bot_action
from models.sniper_bot import sniper_bot_action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='models/model_astro_party.zip')
    parser.add_argument('--opponent', choices=['rusher', 'sniper', 'both'], default='sniper')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--games', type=int, default=1)
    args = parser.parse_args()
    if args.games < 1:
        parser.error('--games must be at least 1')
    torch.set_num_threads(1)
    model = load_model(args.model)
    opponents = {'rusher': rusher_bot_action, 'sniper': sniper_bot_action}
    names = list(opponents) if args.opponent == 'both' else [args.opponent]
    for name in names:
        for seed in range(args.seed, args.seed + args.games):
            print(f'Watching agent (blue) vs {name} (red), seed={seed}', flush=True)
            won, steps = play(model, opponents[name], seed, render=True)
            print(f'Agent won: {won}; steps: {steps}', flush=True)


if __name__ == '__main__':
    main()
