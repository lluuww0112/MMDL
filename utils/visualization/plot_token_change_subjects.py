#!/usr/bin/env python3
"""Show subject composition within paired correct-response length categories."""
from __future__ import annotations
import argparse
from collections import Counter
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
    output = args.output or args.root / 'paired_correct_token_change_subjects.png'
    runs = {}
    for name, (directory, filename) in MODELS.items():
        runs[name] = {}
        for path in sorted((args.root / directory).glob(f'seed*/{filename}')):
            scores, predictions = read_jsonl(path), read_jsonl(path.parent / 'predictions.jsonl')
            if scores.keys() != predictions.keys():
                raise ValueError(f'Prediction and score IDs differ: {path.parent}')
            runs[name][path.parent.name] = (scores, predictions)
    if not runs['Baseline'] or set(runs['Baseline']) != set(runs['KRAdapter']):
        raise ValueError('Models must have matching non-empty seed sets')
    counts = {key: Counter() for key in ('Shorter', 'Same', 'Longer')}
    for seed in sorted(runs['Baseline']):
        bs, bp = runs['Baseline'][seed]
        ks, kp = runs['KRAdapter'][seed]
        if bs.keys() != ks.keys():
            raise ValueError(f'Model question IDs differ: {seed}')
        for key in bs:
            if bs[key]['hit'] == 1 and ks[key]['hit'] == 1:
                subject = bp[key]['subject']
                if subject != kp[key]['subject']:
                    raise ValueError(f'Subject mismatch: {seed}/{key}')
                baseline, adapter = bp[key]['num_output_tokens'], kp[key]['num_output_tokens']
                if baseline <= 0:
                    raise ValueError(f'Non-positive baseline tokens: {seed}/{key}')
                category = 'Shorter' if adapter < baseline else 'Longer' if adapter > baseline else 'Same'
                counts[category][subject] += 1
    labels = list(counts)
    totals = [sum(counts[c].values()) for c in labels]
    if any(n == 0 for n in totals):
        raise ValueError('All three categories must contain at least one response')
    subjects = sorted(set().union(*(set(group) for group in counts.values())))
    matrix = np.array([[counts[c][s] * 100 / total for c, total in zip(labels, totals)]
                       for s in subjects])
    # Split subjects into two panels for a presentation-friendly 16:9 layout.
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5), layout='constrained')
    split = (len(subjects) + 1) // 2
    vmax = matrix.max()
    for ax, start, end in ((axes[0], 0, split), (axes[1], split, len(subjects))):
        subset = subjects[start:end]
        im = ax.imshow(matrix[start:end], cmap='Blues', aspect='auto', vmin=0, vmax=vmax)
        ax.set_xticks(range(3), labels)
        ax.xaxis.tick_top()
        ax.tick_params(axis='x', length=0, labelsize=10, pad=5)
        ax.set_yticks(range(len(subset)), [s.replace('_', ' ') for s in subset], fontsize=8)
        ax.tick_params(axis='y', length=0)
        ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(subset), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1)
        ax.tick_params(which='minor', bottom=False, left=False)
        for row, subject in enumerate(subset):
            for col, category in enumerate(labels):
                value = matrix[start + row, col]
                ax.text(col, row, f'{value:.1f}% ({counts[category][subject]})',
                        ha='center', va='center', fontsize=8,
                        color='white' if value > vmax * .58 else '#243443')
    fig.suptitle('Subject composition by response length change', fontsize=14, fontweight='bold')
    fig.text(.5, -.02,
             f'Shorter n={totals[0]:,}  |  Same n={totals[1]:,}  |  Longer n={totals[2]:,}   '
             '•   Percentages are within each group; counts are question–seed pairs.',
             ha='center', fontsize=9, color='#525c65')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Totals:', dict(zip(labels, totals)))
    print('Column sums:', matrix.sum(axis=0))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
