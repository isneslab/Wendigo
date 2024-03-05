import json
import os
import pickle

import numpy as np
import seaborn as sns

from collections import defaultdict

from matplotlib import pyplot as plt


def set_style():
    sns.set_context('paper')
    sns.set(font='serif')

    sns.set('paper', font='serif', style='ticks', font_scale=1.5, rc={
        'font.family': 'serif',
        'legend.fontsize': 'x-large',
        'xtick.labelsize': 'x-large',
        'ytick.labelsize': 'x-large',
        'axes.labelsize': 'x-large',
        'axes.titlesize': 'x-large',
        'axes.labelpad': 6.0,
        'figure.titlesize': 'x-large',
        'text.usetex': True,
        'figure.figsize': (7.2, 4.45),
        'figure.dpi': 200,
        'savefig.dpi': 200
    })


def style_axes(axes, x_periods=1000, y_periods=200):
    for i, ax in enumerate(axes):
        # Labels
        ax.set_xlabel('Step')
        ax.set_ylabel('Response Time (seconds)')

        # Ticks
        x_tick_range = range(0, x_periods + 1, 25)
        y_tick_range = range(0, y_periods + 1, 10)

        ax.set_xticks(x_tick_range)
        ax.set_yticks(y_tick_range)

        x_labels = [str(x) if x % 100 == 0 else '' for x in x_tick_range]
        ax.set_xticklabels(x_labels)

        # sy_labels = [str(x) if x % 100 == 0 else '' for x in y_tick_range if x < 350]
        # my_labels = ['...']
        # ey_labels = [str(x + 1200) if x % 100 == 0 else '' for x in y_tick_range if x > 350]
        # y_labels = sy_labels + my_labels + ey_labels

        y_labels = [str(x) if x % 20 == 0 else '' for x in y_tick_range]
        ax.set_yticklabels(y_labels)

        ax.tick_params(axis='x', which='major')
        ax.tick_params(axis='y', which='major')

        ax.yaxis.grid(b=True, which='major', color='lightgrey', linestyle='-')

        # Axe limits
        ax.set_xlim(0, x_periods)
        ax.set_ylim(0, y_periods)

        sns.despine(ax=ax, top=True, right=True, bottom=False, left=False)


def add_legend(ax, loc='lower left'):
    lines = ax.get_lines()
    legend = ax.legend(frameon=True, handles=lines, loc=loc, prop={'size': 15})
    legend.get_frame().set_facecolor('#FFFFFF')
    legend.get_frame().set_linewidth(0)
    return legend


def generate_plot(x, y, masks, y_max, labels, colors, markers, save_fname):
    print("Plotting...")

    set_style()

    fig, axes = plt.subplots(1, 1)
    axes = axes if hasattr(axes, '__iter__') else (axes,)

    axes[0].set_title("")

    for i in range(len(x)):
        x_temp = np.array(x[i]).astype(np.double)[masks[i]]
        y_temp = np.array(y[i]).astype(np.double)[masks[i]]

        axes[0].plot(x_temp, y_temp,
                     label=labels[i],
                     alpha=0.7,
                     marker=markers[i],
                     c=colors[i],
                     markeredgewidth=1.5,
                     linewidth=1,
                     markersize=2.5)

    # Legend
    add_legend(axes[0], loc="upper left")

    style_axes(axes, len(x[0]), y_max)
    fig.set_size_inches(9, 4)
    plt.tight_layout()

    plt.savefig(save_fname, transparent=True)


def main():
    os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/universal-darwin'

    tool = ['Wendigo', 'Wendigo', 'Wendigo', 'EvoMaster']
    desc = 'Regular'
    app = 'DVGA'
    agent = ['PPO', 'Random', 'Random-Greedy', 'Black-Box']

    fname = [tool[3] + '-' + app + '-' + agent[3] + '-' + desc + '-Step',
             tool[2] + '-' + app + '-' + agent[2] + '-' + desc + '-Step',
             tool[1] + '-' + app + '-' + agent[1] + '-' + desc + '-Step',
             tool[0] + '-' + app + '-' + agent[0] + '-' + desc + '-Step']

    labels = [tool[3] + ' - ' + agent[3],
              tool[2] + ' - ' + agent[2],
              tool[1] + ' - ' + agent[1],
              tool[0] + ' - ' + agent[0]]

    # colors = ["red", "black", "green", "blue", "green", "red"]
    colors = ["blue", "green", "black", "red", "purple", "green", "red"]

    # markers = ["D", "s", "^", "o", "X", "o"]
    markers = ["o", "^", "s", "D", "X", "o"]

    save_fname = app + '_response_time_graph_' + desc + '.pdf'

    files = ['../steps/paper-results/' + desc + '/' + fname[0] + '-1280-combined.p',
             '../steps/paper-results/' + desc + '/' + fname[1] + '-1280-combined.p',
             '../steps/paper-results/' + desc + '/' + fname[2] + '-1280-combined.p',
             '../steps/paper-results/' + desc + '/' + fname[3] + '-1280-combined.p']

    x = [None for i in files]
    y = [None for i in files]
    masks = [None for i in files]
    y_max = 0
    x_len = 1280

    for i in range(0, len(files)):
        result = pickle.load(open(files[i], 'rb'))

        y[i] = []
        for j in range(0, x_len):
            if result[j][3] or result[j][2] == 0:  # If is rejected or skipped
                y[i] += [None]

            elif result[j][2] > 1600:
                y[i] += [result[j][2]-1200]
                y_max = int(result[j][2]-1200) if int(result[j][2]-1200) > y_max else y_max

            else:
                y[i] += [result[j][2]]
                y_max = int(result[j][2]) if int(result[j][2]) > y_max else y_max

        x[i] = [i+1 for i in range(0, len(y[i]))]
        masks[i] = np.isfinite(np.array(y[i]).astype(np.double))

    y_max = (((y_max // 20) + 1) * 20)

    generate_plot(x=x, y=y, y_max=y_max, masks=masks,
                  labels=labels, colors=colors, markers=markers, save_fname=save_fname)


if __name__ == '__main__':
    main()
