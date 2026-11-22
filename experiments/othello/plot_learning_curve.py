import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / 'learning_curve.json').read_text())
    records = data['checkpoints']
    games = sorted({row['games'] for row in records})
    positions = data['exact_positions']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8.5, 8.3))
    fig.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.095, hspace=0.55)
    fig.text(0.10, 0.95, '4 × 4 Othello self-play', fontsize=20, weight='bold', color='#192939')
    fig.text(0.10, 0.917, 'Points are six-seed means. Shading spans the lowest to highest seed.',
             color='#48576a')

    def series(ax, label, color, metric, include_initial=True):
        checkpoints = games if include_initial else games[1:]
        values = np.array([[metric(row) for row in records if row['games'] == g] for g in checkpoints])
        x = np.log1p(np.array(checkpoints) / 8)
        ax.fill_between(x, values.min(axis=1), values.max(axis=1), color=color, alpha=0.14, lw=0)
        ax.plot(x, values.mean(axis=1), color=color, lw=2.4, marker='o', ms=4.5, label=label)

    score, exact, loss = axes
    for ax, title in zip(axes, ['Match score against the untrained model',
                                'Moves preserving the exact outcome', 'Self-play training loss'], strict=True):
        ax.set_title(title, loc='left', fontsize=13, weight='bold', pad=6)
        ax.spines[['left', 'bottom']].set_color('#b5bec6')
        ax.tick_params(color='#9da9b4')
        ax.grid(axis='y', color='#e4e9ed')
        ax.set_axisbelow(True)
    score.axhline(50, color='#798590', lw=1, ls='--')
    series(score, 'Head-to-head score', '#176d60',
           lambda r: 100 * (r['wins'] + 0.5 * r['draws']) / (r['wins'] + r['draws'] + r['losses']),
           include_initial=False)
    score.set_ylim(39, 73)
    score.set_yticks([40, 50, 60, 70])
    score.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    series(exact, 'Policy alone', '#a24a16', lambda r: 100 * r['policy_correct'] / positions)
    series(exact, 'After 32 simulations', '#286396', lambda r: 100 * r['mcts_correct'] / positions)
    exact.set_ylim(32, 101)
    exact.set_yticks([40, 60, 80, 100])
    exact.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    exact.legend(loc='center right', ncols=2, frameon=False, fontsize=9)
    series(loss, 'Policy loss', '#a24a16', lambda r: r['policy_loss'], include_initial=False)
    series(loss, 'Value loss', '#6b5293', lambda r: r['value_loss'], include_initial=False)
    loss.set_ylim(bottom=0)
    loss.legend(loc='upper right', ncols=2, frameon=False, fontsize=9)
    loss.set_xticks(np.log1p(np.array(games) / 8), [f'{g:,}' for g in games])
    loss.set_xlabel('Self-play games   ·   axis spacing is log(1 + games / 8)', labelpad=8)
    fig.savefig(HERE / 'learning_curve.png', dpi=150, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    main()
