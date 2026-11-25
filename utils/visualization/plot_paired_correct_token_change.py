#!/usr/bin/env python3
"""Plot relative response length changes on questions both models answer correctly."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_correct_response_tokens import ROOT, MODELS, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root / 'paired_correct_token_change_percent.png'
    runs = {}
    for model, (directory, filename) in MODELS.items():
        runs[model] = {}
        for score_path in sorted((args.root / directory).glob(f'seed*/{filename}')):
            scores = read_jsonl(score_path)
            predictions = read_jsonl(score_path.parent / 'predictions.jsonl')
            if scores.keys() != predictions.keys():
                raise ValueError(f'Prediction and score IDs differ: {score_path.parent}')
            runs[model][score_path.parent.name] = (scores, predictions)
    if not runs['Baseline'] or set(runs['Baseline']) != set(runs['KRAdapter']):
        raise ValueError('Models must have the same non-empty set of seeds')
    changes, seed_counts = [], {}
    for seed in sorted(runs['Baseline']):
        bs, bp = runs['Baseline'][seed]
        ks, kp = runs['KRAdapter'][seed]
        if bs.keys() != ks.keys():
            raise ValueError(f'Model question IDs differ for {seed}')
        values = []
        for i in bs:
            if bs[i]['hit'] == 1 and ks[i]['hit'] == 1:
                base = bp[i]['num_output_tokens']
                if base <= 0:
                    raise ValueError(f'Baseline token count must be positive: {seed}/{i}')
                values.append((kp[i]['num_output_tokens'] - base) / base * 100)
        if not values:
            raise ValueError(f'No shared correct responses in {seed}')
        changes.extend(values)
        seed_counts[seed] = len(values)
    changes = np.asarray(changes)
    shorter = int(np.sum(changes < 0)); same = int(np.sum(changes == 0)); longer = int(np.sum(changes > 0))
    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    ax.pie([shorter, same, longer], labels=['Shorter', 'Same length', 'Longer'],
           colors=['#59758f', '#b7c3ca', '#e28243'], startangle=90,
           autopct=lambda pct: f'{pct:.1f}%\n(n={round(pct * len(changes) / 100):,})',
           pctdistance=.72, labeldistance=1.08,
           wedgeprops={'edgecolor': 'white', 'linewidth': 2},
           textprops={'fontsize': 12})
    ax.set_title('KRAdapter response length relative to baseline', fontsize=15,
                 fontweight='bold', pad=20)
    fig.text(.5, .02, f'{len(changes):,} question–seed pairs answered correctly by both models',
             ha='center', fontsize=10, color='#525c65')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Counts by seed:', seed_counts)
    print(f'n={len(changes)}, shorter={shorter}, same={same}, longer={longer}, median={np.median(changes):.2f}%')
    print('Changes >= 200%:', int(np.sum(changes >= 200)))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
