#!/usr/bin/env python3
"""Compare subject composition of shorter and longer paired correct responses."""
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
    output = args.output or args.root / 'paired_correct_shorter_longer_subject_pies.png'
    runs = {}
    for name, (directory, filename) in MODELS.items():
        runs[name] = {}
        for path in sorted((args.root / directory).glob(f'seed*/{filename}')):
            scores = read_jsonl(path)
            predictions = read_jsonl(path.parent / 'predictions.jsonl')
            if scores.keys() != predictions.keys():
                raise ValueError(f'Prediction and score IDs differ: {path.parent}')
            runs[name][path.parent.name] = scores, predictions
    if not runs['Baseline'] or set(runs['Baseline']) != set(runs['KRAdapter']):
        raise ValueError('Models must have matching non-empty seed sets')
    # Rank by mean per-seed subject accuracy improvement, including cutoff ties.
    accuracy = {}
    for model, model_runs in runs.items():
        per_subject = {}
        for scores, _ in model_runs.values():
            hits, totals_by_subject = Counter(), Counter()
            for score in scores.values():
                subject = score['subject']
                hits[subject] += score['hit']
                totals_by_subject[subject] += 1
            for subject, total in totals_by_subject.items():
                per_subject.setdefault(subject, []).append(hits[subject] / total)
        accuracy[model] = {subject: np.mean(values) for subject, values in per_subject.items()}
    if accuracy['Baseline'].keys() != accuracy['KRAdapter'].keys():
        raise ValueError('Models must have matching subjects')
    improvement = {subject: accuracy['KRAdapter'][subject] - value
                   for subject, value in accuracy['Baseline'].items()}
    positive = sorted((v for v in improvement.values() if v > 1e-10), reverse=True)
    cutoff = positive[min(4, len(positive) - 1)] if positive else float('inf')
    highlighted = {subject for subject, value in improvement.items()
                   if value > 1e-10 and value >= cutoff - 1e-10}
    print('Highlighted (accuracy pp):',
          {s: round(improvement[s] * 100, 2) for s in sorted(highlighted)})
    counts = {'Shorter': Counter(), 'Same': Counter(), 'Longer': Counter()}
    for seed in sorted(runs['Baseline']):
        bs, bp = runs['Baseline'][seed]
        ks, kp = runs['KRAdapter'][seed]
        if bs.keys() != ks.keys():
            raise ValueError(f'Model question IDs differ: {seed}')
        for key in bs:
            if bs[key]['hit'] != 1 or ks[key]['hit'] != 1:
                continue
            subject = bp[key]['subject']
            if subject != kp[key]['subject']:
                raise ValueError(f'Subject mismatch: {seed}/{key}')
            baseline, adapter = bp[key]['num_output_tokens'], kp[key]['num_output_tokens']
            if baseline <= 0:
                raise ValueError(f'Non-positive baseline tokens: {seed}/{key}')
            if adapter < baseline:
                counts['Shorter'][subject] += 1
            elif adapter > baseline:
                counts['Longer'][subject] += 1
            else:
                counts['Same'][subject] += 1
    subjects = sorted(set().union(*(set(group) for group in counts.values())))
    totals = {name: sum(counts[name].values()) for name in counts}
    if not subjects or not totals['Shorter'] or not totals['Longer']:
        raise ValueError('Shorter and Longer categories must both contain responses')
    colors = list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('tab20b').colors)
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.06, 1], hspace=.06, wspace=.15)
    for j, name in enumerate(('Shorter', 'Longer')):
        ax = fig.add_subplot(grid[0, j])
        ax.pie([counts[name][s] for s in subjects], colors=colors[:len(subjects)],
               startangle=90, counterclock=False,
               wedgeprops={'width': .46, 'edgecolor': 'white', 'linewidth': .5})
        ax.text(0, .09, name, ha='center', va='center', fontsize=15, fontweight='bold')
        ax.text(0, -.12, f'n = {totals[name]:,}', ha='center', va='center', fontsize=11)
    key_ax = fig.add_subplot(grid[0, 2]); key_ax.axis('off')
    key_ax.text(0, .8, 'Subject share by category', fontsize=15, fontweight='bold')
    key_ax.text(0, .68, 'Each ring totals 100%.\nSame-length pairs are excluded.\nThe table shows Short / Same / Long\nas percentages of each subject’s\nshared correct responses.', fontsize=11, linespacing=1.6, va='top')
    blocks = np.array_split(np.arange(len(subjects)), 3)
    for block_index, indices in enumerate(blocks):
        ax = fig.add_subplot(grid[1, block_index]); ax.set_axis_off()
        ax.text(.04, 1.0, 'SUBJECT', weight='bold', fontsize=9, transform=ax.transAxes)
        ax.text(.70, 1.0, 'SHORT', weight='bold', fontsize=9, ha='right', transform=ax.transAxes)
        ax.text(.85, 1.0, 'SAME', weight='bold', fontsize=9, ha='right', transform=ax.transAxes)
        ax.text(1.0, 1.0, 'LONG', weight='bold', fontsize=9, ha='right', transform=ax.transAxes)
        for row, index in enumerate(indices):
            subject = subjects[index]; y = .94 - row * .092
            ax.add_patch(plt.Rectangle((.01, y - .012), .025, .025, transform=ax.transAxes,
                                       facecolor=colors[index], edgecolor='none'))
            ax.text(.05, y, subject.replace('_', ' '), fontsize=6.6 if len(subject) > 30 else 8.0, va='center', transform=ax.transAxes,
                    color='#bb3344' if subject in highlighted else '#222222',
                    fontweight='bold' if subject in highlighted else 'normal')
            for x, name in ((.70, 'Shorter'), (.85, 'Same'), (1.0, 'Longer')):
                ax.text(x, y, f'{counts[name][subject] / sum(group[subject] for group in counts.values()) * 100:.1f}%',
                        fontsize=8.5, ha='right', va='center', transform=ax.transAxes)
    fig.suptitle('Subject composition of shorter and longer responses', fontsize=17, fontweight='bold', y=.98)
    fig.text(.5, .025, 'Table: per-subject percentages across Short / Same / Long; rings: subject composition within each group.\n'
             'Red: top five accuracy gains vs baseline (including ties), averaged across seeds.',
             ha='center', fontsize=10, color='#525c65')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, facecolor='white')
    plt.close(fig)
    print('Totals:', totals, 'subjects:', len(subjects))
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
