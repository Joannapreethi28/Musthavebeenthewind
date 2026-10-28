"""Four scientific figure plates, saved as shareable PNG and offline SVG."""
import os
from common import ROOT, RESULTS
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

PAPER, INK, RED, BLUE, MUTED = '#F3EFE6', '#1B1A17', '#B83216', '#4C5C6B', '#68645C'
NAMES = {'persistence': 'Persistence', 'persistence_diurnal': 'Persistence + diurnal', 'lstm': 'LSTM'}


def setup() -> None:
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'figure.facecolor': PAPER,
                         'axes.facecolor': PAPER, 'axes.edgecolor': '#CFC8B8', 'text.color': INK,
                         'axes.labelcolor': INK, 'xtick.color': BLUE, 'ytick.color': BLUE,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'grid.color': '#CFC8B8', 'grid.linewidth': .5, 'svg.fonttype': 'path',
                         'savefig.facecolor': PAPER})


def save(fig, name: str) -> None:
    folder = RESULTS / 'figs'
    folder.mkdir(exist_ok=True)
    for suffix in ('png', 'svg'):
        fig.savefig(folder / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def flux_axis(axis) -> None:
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: rf'$10^{{{y:g}}}$'))
    axis.set_ylabel('Electron flux [e cm⁻² s⁻¹ sr⁻¹]')


def storm_figure(data: dict):
    fig, ax = plt.subplots(figsize=(10.2, 4.0), layout='constrained')
    times = pd.to_datetime(data['storm']['time'])
    for key, color, width, style in [('persistence_diurnal', MUTED, 1.0, '--'), ('obs', BLUE, 1.25, '-'), ('lstm', RED, 1.25, '-')]:
        values = np.array([np.nan if v is None else v for v in data['storm'][key]])
        ax.plot(times, values, color=color, lw=width, ls=style)
    endpoints = []
    for key, label, color in [('obs', 'Observed', BLUE), ('lstm', 'LSTM · seed 0', RED), ('persistence_diurnal', 'Persistence\n+ diurnal', MUTED)]:
        values = np.array([np.nan if v is None else v for v in data['storm'][key]])
        last = np.flatnonzero(np.isfinite(values))[-1]
        endpoints.append((values[last], times[last], label, color))
    previous_y = -np.inf
    for value, stamp, label, color in sorted(endpoints):
        label_y = max(value, previous_y + .45)
        ax.annotate(label, xy=(stamp, value), xytext=(times[-1] + pd.Timedelta(days=1), label_y),
                    color=color, fontsize=9, va='center', arrowprops={'arrowstyle': '-', 'color': color, 'lw': .6})
        previous_y = label_y
    storm_time = pd.Timestamp('2017-09-08', tz='UTC')
    ax.axvline(storm_time, color=INK, lw=.75, ls=':')
    ax.text(storm_time, .02, ' September storm interval', transform=ax.get_xaxis_transform(), fontsize=8, rotation=90, va='bottom')
    ax.set_xticks(pd.date_range(times[0], times[-1], freq='4D'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.set_xlim(times[0], times[-1] + pd.Timedelta(days=4.4))
    ax.grid(axis='y')
    flux_axis(ax)
    ax.set_xlabel('Target hour start · UTC · gaps indicate unscored hours')
    return fig


def render_figures(data: dict) -> None:
    setup()
    save(storm_figure(data), '01_storm')
    metrics = data['metrics']
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), layout='constrained')
    for ax, split, title in zip(axes, ['val', 'test'], ['Validation · 2016', 'Held-out test · 2017']):
        baseline = metrics['models']['persistence_diurnal'][split]['pe']
        ax.axvline(baseline, color=MUTED, lw=1, ls='--')
        for row, name in enumerate(NAMES):
            value = metrics['models'][name][split]
            mean = value['aggregate']['pe']['mean'] if name == 'lstm' else value['pe']
            std = value['aggregate']['pe']['std'] if name == 'lstm' else 0
            ax.errorbar(mean, 2 - row, xerr=std, fmt='o' if name == 'lstm' else 's', color=RED if name == 'lstm' else BLUE, capsize=4, ms=7)
        ax.set_yticks([2, 1, 0], NAMES.values())
        ax.set_ylim(-.6, 2.6)
        ax.set_xlim(0, 1)
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_xlabel('Prediction efficiency · higher is better')
        ax.grid(axis='x')
    save(fig, '02_scoreboard')
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), layout='constrained')
    lt = np.array(data['diurnal']['lt']) + .5
    axes[0].plot(lt, data['diurnal']['mean_logflux'], color=BLUE, marker='.', lw=1.5)
    axes[0].set_title('Daily cycle · geometric mean training flux', loc='left', fontsize=12)
    flux_axis(axes[0])
    for name, color, style in [('persistence', '#999183', ':'), ('persistence_diurnal', BLUE, '--'), ('lstm', RED, '-')]:
        vals = data['diurnal']['err_by_lt'][name]
        axes[1].plot(lt, vals, color=color, ls=style, lw=1.5)
        axes[1].text(24.1, vals[-1], NAMES[name], color=MUTED if name == 'persistence' else color, fontsize=8, va='center')
    axes[1].set_title('Test error · target local time', loc='left', fontsize=12)
    axes[1].set_ylabel('RMSE [log₁₀ flux] · LSTM seed 0')
    axes[1].set_xlim(0, 37)
    for ax in axes:
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel('Local solar time [hours]')
        ax.grid(axis='y')
    save(fig, '03_diurnal')
    fig = plt.figure(figsize=(10.2, 4.3), layout='constrained')
    grid = fig.add_gridspec(2, 3, height_ratios=[.5, 2.1])
    timeline = fig.add_subplot(grid[0, :])
    for start, width, label, color in [(0, 4, 'TRAIN  2012–2015', BLUE), (4, 1, 'VAL  2016', MUTED), (5, 1, 'TEST  2017', RED)]:
        timeline.barh(0, width, left=start, height=.8, color=color, alpha=.18)
        timeline.text(start + width / 2, 0, label, ha='center', va='center', fontsize=9)
    timeline.set_xlim(0, 6)
    timeline.axis('off')
    for seed in range(3):
        ax = fig.add_subplot(grid[1, seed])
        history = data['history'][f'lstm_seed{seed}']
        ax.plot(history['epoch'], history['train_mse'], color=BLUE, lw=1, label='Train')
        ax.plot(history['epoch'], history['val_mse'], color=RED, lw=1.5, label='Validation')
        best = metrics['runs'][seed]['best_epoch']
        ax.axvline(best, color=MUTED, ls=':', lw=1)
        ax.set_title(f'Seed {seed} · best epoch {best}', loc='left', fontsize=10)
        ax.set_xlabel('Epoch')
        if seed == 0:
            ax.set_ylabel('MSE on Δ log₁₀ flux')
        ax.text(.95, .96, 'Validation', color=RED, ha='right', va='top', transform=ax.transAxes, fontsize=8)
        ax.text(.95, .86, 'Train', color=BLUE, ha='right', va='top', transform=ax.transAxes, fontsize=8)
        ax.grid(axis='y')
    save(fig, '04_method')
    print('PASS four figures saved as PNG and SVG.', flush=True)
