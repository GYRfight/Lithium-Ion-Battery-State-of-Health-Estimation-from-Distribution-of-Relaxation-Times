"""Plot only grouped RMSE and MAE charts from the three prediction CSVs.

Usage:
    python plot_benchmark_grouped.py

Dependencies: numpy, pandas, matplotlib
Outputs: grouped_RMSE_MAE.png and grouped_RMSE_MAE.pdf, two panels in one row.
Place this script beside own.csv, LCT.csv, and 6cell.csv.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


# Edit these settings to adjust the appearance.
ROOT = Path(__file__).resolve().parent
FIGSIZE = (7.09, 3.35)  # Approximately 18 cm wide.
DPI = 300
FONT_SIZE = 8
SHOW_VALUES = True
SHOW_WINDOW = True
BAR_WIDTH = 0.09
GROUP_GAP = 0.09  # Gap between dataset groups; decrease for a tighter layout.
DATASETS = [('Dataset 1', 'own.csv'),
            ('Dataset 2', 'LCT.csv'),
            ('Dataset 3', '6cell.csv')]

# Method, display name, color, hatch. Identical methods retain their style.
METHODS = [
    ('DRA',              'DRA',              '#D47B59', ''),
    ('DRT_Ridge',        'DRT-Ridge',        '#95ACC3', '//'),
    ('DRT_CNN',          'DRT-CNN',          '#B6A7C5', '\\\\'),
    ('Handcrafted DRT',  'Handcrafted DRT',  '#A4BBAE', '..'),
    ('ECM_RandomForest', 'ECM-RF',           '#C7ADA0', 'xx'),
    ('EIS_ExtraTrees',   'EIS-ExtraTrees',   '#96BEC5', '--'),
    ('EIS_CNN',          'EIS-CNN',          '#D2C18D', '++'),
    ('DRT_Transformer',  'DRT-Transformer',  '#ADB88B', '**'),
]


def read_metrics():
    """Pool all provided predictions; do not average per-cell RMSE values."""
    known = {method for method, *_ in METHODS}
    rows = []
    for dataset, filename in DATASETS:
        frame = pd.read_csv(ROOT / filename)
        if frame[['Cell', 'Cycle', 'SOH_true']].isna().any().any():
            raise ValueError(f'{filename}: missing identifiers or true values')
        if frame.duplicated(['Cell', 'Cycle']).any():
            raise ValueError(f'{filename}: repeated cell-cycle keys')

        truth = frame['SOH_true'].to_numpy(dtype=float)
        if not np.isfinite(truth).all() or not ((truth > 0) & (truth < 1.5)).all():
            raise ValueError(f'{filename}: expected finite SOH fractions')

        columns = frame.columns[frame.columns.get_loc('SOH_true') + 1:]
        for method in columns:
            base, _, suffix = method.rpartition('.')
            if suffix.isdigit() and base in frame:
                if not np.array_equal(frame[method], frame[base]):
                    raise ValueError(f'{filename}: conflicting duplicate {method}')
                continue  # Identical duplicate prediction columns count once.
            if method not in known:
                raise ValueError(f'Add the new method to METHODS: {method}')

            prediction = frame[method].to_numpy(dtype=float)
            if not np.isfinite(prediction).all():
                raise ValueError(f'{filename}: missing/nonfinite {method} predictions')
            error = 100 * (prediction - truth)
            rows.append({'Dataset': dataset, 'Method': method,
                         'RMSE': np.sqrt(np.mean(error ** 2)),
                         'MAE': np.mean(np.abs(error))})
    return pd.DataFrame(rows)


def draw_panel(ax, results, metric, panel_label):
    maximum = results[metric].max()
    groups = []
    cursor = 0.0
    centers = []
    for dataset, _ in DATASETS:
        values = results.loc[results.Dataset == dataset].set_index('Method')
        available = [entry for entry in METHODS if entry[0] in values.index]
        group_width = len(available) * BAR_WIDTH
        center = cursor + group_width / 2
        centers.append(center)
        groups.append((center, values, available))
        cursor += group_width + GROUP_GAP

    for center, values, available in groups:
        offsets = (np.arange(len(available)) - (len(available) - 1) / 2) * BAR_WIDTH
        for offset, (method, label, color, hatch) in zip(offsets, available):
            value = values.loc[method, metric]
            x = center + offset
            ax.bar(x, value, width=BAR_WIDTH * 0.88, color=color,
                   edgecolor='#555555', linewidth=0.5, hatch=hatch, zorder=3)
            if SHOW_VALUES:
                ax.annotate(f'{value:.2f}', (x, value), xytext=(0, 4),
                            textcoords='offset points', rotation=90,
                            rotation_mode='anchor', ha='left', va='center',
                            fontsize=6.5, fontweight='bold' if method == 'DRA' else 'normal')

    ax.set_xticks(centers, [name for name, _ in DATASETS])
    ax.set_xlim(-BAR_WIDTH * 0.3, cursor - GROUP_GAP + BAR_WIDTH * 0.3)
    ax.set_ylim(0, maximum * 1.25)
    ax.set_ylabel(metric, fontsize=9)  # Values remain in percentage points.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', length=3)
    ax.grid(False)

    ax.text(-0.14, 1.03, panel_label, transform=ax.transAxes,
            fontsize=9, fontweight='bold', ha='left', va='bottom')


def plot_grouped(results):
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.15,
                        top=0.79, wspace=0.22)
    draw_panel(axes[0], results, 'RMSE', '(a)')
    draw_panel(axes[1], results, 'MAE', '(b)')

    present = set(results.Method)
    handles = [Patch(facecolor=color, edgecolor='#555555', linewidth=0.5,
                     hatch=hatch, label=label)
               for method, label, color, hatch in METHODS if method in present]
    # Two legend rows (5 + 4 entries), read left-to-right in method order.
    # Matplotlib fills legend columns vertically, so interleave the two rows.
    columns = (len(handles) + 1) // 2
    handles = [handles[i] for column in range(columns)
               for i in (column, column + columns) if i < len(handles)]
    # One legend shared by the two panels.
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.995),
               ncol=columns, frameon=False, fontsize=7.5, handlelength=2.0,
               columnspacing=1.2, labelspacing=0.7)

    # Preserve the different Transformer names in the supplied datasets.
    # Each bar pools all supplied predictions; no repeated-run error bars.
    destination = ROOT / 'grouped_RMSE_MAE.png'
    fig.savefig(destination, dpi=DPI, facecolor='white')
    print(f'Saved: {destination}')
    pdf_destination = ROOT / 'grouped_RMSE_MAE.pdf'
    fig.savefig(pdf_destination, format='pdf', facecolor='white')
    print(f'Saved: {pdf_destination}')
    return fig


def main():
    plt.rcParams.update({'font.family': 'Arial', 'font.size': FONT_SIZE,
                         'axes.linewidth': 0.8, 'hatch.linewidth': 0.45,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    results = read_metrics()
    plot_grouped(results)
    if SHOW_WINDOW:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    main()
