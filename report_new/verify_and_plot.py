"""Check saved report numbers and redraw the required figure; no training runs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / 'report_work/results'

def read(name):
    return json.loads((DATA / name).read_text())

fig, axes = plt.subplots(1, 2, figsize=(8, 2.65), sharex=True, sharey=True)
for ax, label in zip(axes, ['dqn', 'ddqn']):
    data = read(f'mc_{label}.json')
    assert data['rollouts'] == 64 and len(data['states']) == 12
    gaps = []
    for row in data['states']:
        returns = np.array(row['returns'])
        assert len(returns) == 64 and not row['capped']
        assert np.isclose(returns.mean(), row['mc'])
        assert np.isclose(1.998340543 * returns.std(ddof=1) / 8, row['ci95_half'])
        gaps.append(row['predicted'] - row['mc'])
    assert np.isclose(np.mean(gaps), data['mean_bias'])
    rows = data['states']
    ax.errorbar([r['mc'] for r in rows], [r['predicted'] for r in rows],
                xerr=[r['ci95_half'] for r in rows], fmt='o', markersize=4,
                capsize=2, color='#205d88', linewidth=1)
    ax.plot([-700, 500], [-700, 500], '--', color='gray', linewidth=1)
    ax.set(title=label.upper() + ', 40,000 steps', xlabel='Monte Carlo Q (95% CI)')
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    print(label, 'verified mean gap:', round(np.mean(gaps), 1))
axes[0].set_ylabel('Predicted Q')
fig.tight_layout()
fig.savefig(HERE / 'mc_comparison.png', dpi=220)
plt.close(fig)

rows = read('astro_algorithm_comparison.json')
assert len(rows) == 6
for row in rows:
    assert row['steps'] == 8192
    for opponent in ['rusher', 'sniper']:
        games = [g for g in row['episodes'] if g['opponent'] == opponent]
        assert sorted(g['seed'] for g in games) == list(range(72000, 72030))
        wins = sum(g['win'] for g in games)
        assert np.isclose(wins / 30, row['metrics'][f'eval/{opponent}_win_rate'])
        print(row['algorithm'], row['seed'], opponent, f'{wins}/30')
for algorithm in ['PPO', 'A2C']:
    values = [r['metrics']['eval/sniper_win_rate'] for r in rows if r['algorithm'] == algorithm]
    print(algorithm, 'mean Sniper:', round(np.mean(values) * 100, 1))

final = read('2048_final.json')
scores = [r['score'] for r in final['evaluation']['rows']]
assert len(scores) == 100 and np.isclose(np.mean(scores), 1238)
assert max(r['highest'] for r in final['evaluation']['rows']) == 512
print('All saved-data checks passed. This does not rerun training or evaluation.')
