import sys
import os
import pandas
import matplotlib
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
import scipy.stats

# from llrl.utils.utils import mean_confidence_interval
# from llrl.utils.save import csv_path_from_agent

COLOR_SHIFT = 0
FONT_SIZE = 15

'''
COLOR_LIST = [
    [118, 167, 125],
    [102, 120, 173],
    [198, 113, 113],
    [94, 94, 94],
    [169, 193, 213],
    [230, 169, 132],
    [192, 197, 182],
    [210, 180, 226],
    [167, 167, 125],
    [125, 167, 125]
]
'''
COLOR_LIST = [[102, 120, 173], [240, 163, 255], [113, 198, 113],\
                [85, 85, 85], [198, 113, 113],\
                [142, 56, 142], [125, 158, 192],[184, 221, 255],\
                [153, 63, 0], [142, 142, 56], [56, 142, 142]]
_COLOR_LIST = [[102, 120, 173], [240, 163, 255], [113, 198, 113],\
                [85, 85, 85], [198, 113, 113],\
                [142, 56, 142], [125, 158, 192],[184, 221, 255],\
                [153, 63, 0], [142, 142, 56], [56, 142, 142]]
# COLOR_LIST = [  # Average
#     [153, 194, 255],

#     [159, 198, 177],
#     [128, 179, 151],
#     # [96, 160, 126],

#     [90, 90, 90],

#     [255, 166, 77],
#     [255, 102, 102]
# ]

# _COLOR_LIST = [  # Custom plot
#     [153, 194, 255],

#     [255, 161, 102],  # [128, 179, 151],
#     [255, 102, 102],  # [96, 160, 126],

#     [128, 128, 128],
#     [90, 90, 90]
# ]

def mean_confidence_interval(data, confidence=0.95):
    """
    Compute the mean and confidence interval of the the input data array-like.
    :param data: (array-like)
    :param confidence: probability of the mean to lie in the interval
    :return: (tuple) mean, interval upper-endpoint, interval lower-endpoint
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    if confidence is None:
        return m, None, None
    else:
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

def csv_path_from_agent(root_path, agent):
    """
    Get the saving path from agent object and root path.
    :param root_path: (str)
    :param agent: (object)
    :return: (str)
    """
    return root_path + agent.get_name() + '_result.csv'


def unzip(lst):
    x, x_lo, x_up = [], [], []
    for e in lst:
        tmp = list(zip(*e))
        x.append(np.array(tmp[0]))
        x_lo.append(np.array(tmp[1]))
        x_up.append(np.array(tmp[2]))
    return x, x_lo, x_up


def averaged_lifelong_plot(
        dfs,
        agents,
        path,
        n_tasks,
        n_episodes,
        confidence,
        open_plot,
        plot_title,
        norm_ag=None,  # normalize everything w.r.t. the agent of the specified index
        which_norm_ag=0,  # 0: normalize everything; 1: normalize w.r.t. episodes; 2: normalize w.r.t. tasks
        plot_legend=0,
        legend_at_bottom=False,
        episodes_moving_average=False,
        episodes_ma_width=10,
        tasks_moving_average=False,
        tasks_ma_width=10,
        latex_rendering=False,
        figure_title=None):

    # Extract data
    n_agents = len(agents)
    tre, dre, trt, drt = [], [], [], []
    for i in range(n_agents):
        tre_i, dre_i = [], []
        for j in range(1, n_episodes + 1):
            tr_norm, dr_norm = 1., 1.
            if norm_ag is not None and which_norm_ag in [0, 1]:
                df = dfs[norm_ag]
                # TODO set this param (1500 for tight and 12 for corridor)
                df = df.loc[df['episode'] >= 12]  # remove extra episodes
                df = df.loc[df['episode'] <= n_episodes]  # remove extra episodes
                df = df.loc[df['task'] <= n_tasks]  # remove extra tasks
                tr_norm = max(df['return'].mean(), .001)
                dr_norm = max(df['discounted_return'].mean(), .001)
                
            df = dfs[i].loc[dfs[i]['episode'] == j]  # data-frame for agent i, episode j
            df = df.loc[df['task'] <= n_tasks]  # only select tasks <= n_tasks

            tre_i.append(mean_confidence_interval(df['return'] / tr_norm, confidence))
            dre_i.append(mean_confidence_interval(df['discounted_return'] / dr_norm, confidence))
        tre.append(tre_i)
        dre.append(dre_i)

        trt_i, drt_i = [], []
        for j in range(1, n_tasks + 1):
            tr_norm, dr_norm = 1., 1.
            if norm_ag is not None and which_norm_ag in [0, 2]:
                df = dfs[norm_ag]
                df = df.loc[df['episode'] <= n_episodes]  # remove extra episodes
                df = df.loc[df['task'] <= n_tasks]  # remove extra tasks
                df = df.loc[df['task'] == j ]  # remove extra tasks
                tr_norm = max(df['return'].mean(), .001)
                dr_norm = max(df['discounted_return'].mean(), .001)

            df = dfs[i].loc[dfs[i]['task'] == j]  # data-frame for agent i, task j
            df = df.loc[df['episode'] <= n_episodes]  # only select episodes <= n_episodes

            trt_i.append(mean_confidence_interval(df['return'] / tr_norm, confidence))
            drt_i.append(mean_confidence_interval(df['discounted_return'] / dr_norm, confidence))
        trt.append(trt_i)
        drt.append(drt_i)

    # x-axis
    x_e = np.array(range(1, n_episodes + 1))
    x_t = np.array(range(1, n_tasks + 1))

    # Unzip everything for confidence intervals
    tre, tre_lo, tre_up = unzip(tre)
    dre, dre_lo, dre_up = unzip(dre)
    trt, trt_lo, trt_up = unzip(trt)
    drt, drt_lo, drt_up = unzip(drt)

    # Labels
    x_label_e = r'Episode number'
    x_label_t = r'Task number'
    if norm_ag is None:
        y_labels = [r'Average Reward', r'Average Discounted Reward', r'Average Reward', r'Average Discounted Reward']
    else:
        y_labels = [
            r'Average Relative Return' if
            which_norm_ag in [0, 1] else r'Average Return',
            r'Average Relative Discounted Return' if
            which_norm_ag in [0, 1] else r'Average Discounted Return',
            r'Average Relative Return' if
            which_norm_ag in [0, 2] else r'Average Return',
            r'Average Relative Discounted Return' if
            which_norm_ag in [0, 2] else r'Average Discounted Return'
        ]

    # Plots w.r.t. episodes
    _lgd = True if plot_legend in [1, 3] else False
    plot(path, pdf_name='return_vs_episode', agents=agents, x=x_e, y=tre, y_lo=tre_lo, y_up=tre_up,
         x_label=x_label_e, y_label=y_labels[0], title_prefix=r'Average Reward: ', open_plot=open_plot,
         plot_title=plot_title, plot_legend=_lgd, legend_at_bottom=legend_at_bottom,
         ma=episodes_moving_average, ma_width=episodes_ma_width, latex_rendering=latex_rendering,
         x_cut=None, plot_markers=True, figure_title=figure_title, marke_every=3)
    plot(path, pdf_name='discounted_return_vs_episode', agents=agents, x=x_e, y=dre, y_lo=dre_lo, y_up=dre_up,
         x_label=x_label_e, y_label=y_labels[1], title_prefix=r'Average Discounted Reward: ',
         open_plot=open_plot, plot_title=plot_title, plot_legend=_lgd, legend_at_bottom=legend_at_bottom,
         ma=episodes_moving_average, ma_width=episodes_ma_width, latex_rendering=latex_rendering,
         x_cut=None, plot_markers=True, figure_title=figure_title, marke_every=3)

    # Plots w.r.t. tasks
    # _lgd = True if plot_legend in [2, 3] else False
    _lgd = True
    _lgd_btm = False
    _cst = False
    plot(path, pdf_name='return_vs_task', agents=agents, x=x_t, y=trt, y_lo=trt_lo, y_up=trt_up,
         x_label=x_label_t, y_label=y_labels[2], title_prefix=r'Average Reward: ', open_plot=open_plot,
         plot_title=plot_title, plot_legend=_lgd, legend_at_bottom=_lgd_btm, figure_title=figure_title,
         ma=tasks_moving_average, ma_width=tasks_ma_width, latex_rendering=latex_rendering, custom=_cst)
    plot(path, pdf_name='discounted_return_vs_task', agents=agents, x=x_t, y=drt, y_lo=drt_lo, y_up=drt_up,
         x_label=x_label_t, y_label=y_labels[3], title_prefix=r'Average Discounted Reward: ',
         open_plot=open_plot, plot_title=plot_title, plot_legend=_lgd, legend_at_bottom=_lgd_btm,
         figure_title=figure_title,
         ma=tasks_moving_average, ma_width=tasks_ma_width, latex_rendering=latex_rendering, custom=_cst)


def raw_lifelong_plot(
        dfs,
        agents,
        path,
        n_tasks,
        n_episodes,
        confidence=None,
        open_plot=False,
        plot_title=True,
        plot_legend=True,
        legend_at_bottom=False,
        ma=False,
        ma_width=10,
        latex_rendering=False):

    x = np.array(range(1, n_episodes + 1))
    x_label = r'Episode number'
    labels = ['Task ' + str(t) for t in range(1, n_tasks + 1)]
    for i in range(len(agents)):
        tr_per_task, tr_per_task_lo, tr_per_task_up = [], [], []
        dr_per_task, dr_per_task_lo, dr_per_task_up = [], [], []
        for j in range(1, n_tasks + 1):
            task_j = dfs[i].loc[dfs[i]['task'] == j]
            # n_instances = task_j['instance'].nunique()
            tr, tr_lo, tr_up = [], [], []
            dr, dr_lo, dr_up = [], [], []
            for k in range(1, n_episodes + 1):
                task_j_episodes_k = task_j.loc[task_j['episode'] == k]
                tr_mci = mean_confidence_interval(task_j_episodes_k['return'], confidence)
                dr_mci = mean_confidence_interval(task_j_episodes_k['discounted_return'], confidence)
                tr.append(tr_mci[0])
                tr_lo.append(tr_mci[1])
                tr_up.append(tr_mci[2])
                dr.append(dr_mci[0])
                dr_lo.append(dr_mci[1])
                dr_up.append(dr_mci[2])
            tr_per_task.append(tr)
            tr_per_task_lo.append(tr_lo)
            tr_per_task_up.append(tr_up)
            dr_per_task.append(dr)
            dr_per_task_lo.append(dr_lo)
            dr_per_task_up.append(dr_up)
        agent_name = str(agents[i])
        pdf_name = 'lifelong-' + agent_name
        pdf_name = pdf_name.lower()

        plot_color_bars(path, pdf_name=pdf_name+'-return', x=x, y=tr_per_task, y_lo=None, y_up=None, cb_min=1,
                        cb_max=n_tasks + 1, cb_step=1, x_label=x_label,
                        y_label='Return', title_prefix='', labels=labels, cbar_label='Task number',
                        title=agent_name, plot_title=plot_title, plot_markers=False, plot_legend=False,
                        legend_at_bottom=legend_at_bottom, ma=ma, ma_width=ma_width,
                        latex_rendering=latex_rendering)

        plot_color_bars(path, pdf_name=pdf_name+'-discounted-return', x=x, y=dr_per_task, y_lo=None, y_up=None, cb_min=1,
                        cb_max=n_tasks + 1, cb_step=1, x_label=x_label,
                        y_label='Discounted return', title_prefix='', labels=labels, cbar_label='Task number',
                        title=agent_name, plot_title=plot_title, plot_markers=False, plot_legend=False,
                        legend_at_bottom=legend_at_bottom, ma=ma, ma_width=ma_width,
                        latex_rendering=latex_rendering)


def custom_label(agent, task):
    return str(agent) + r' Task = ' + str(task)


def custom_lifelong_plot(dfs, agents, path, n_tasks, n_episodes):
    rmax_id = 0
    lrmax_id = 3
    maxqinit_id = 4
    x = np.array((range(1, n_episodes + 1)))
    y, labels = [], []

    # Select data subset
    for i in range(len(agents)):
        dfs[i] = dfs[i].loc[dfs[i]['instance'] == 0]
        dfs[i] = dfs[i].loc[dfs[i]['task'] <= n_tasks]
        dfs[i] = dfs[i].loc[dfs[i]['episode'] <= n_episodes]

    # Normalizers
    df = dfs[rmax_id]
    se = 1500
    norm_1 = df.loc[df['task'] == 1].loc[df['episode'] >= se]['discounted_return'].mean()
    norm_2 = df.loc[df['task'] == 2].loc[df['episode'] >= se]['discounted_return'].mean()
    norm_11 = df.loc[df['task'] == 11].loc[df['episode'] >= se]['discounted_return'].mean()
    norm_12 = df.loc[df['task'] == 12].loc[df['episode'] >= se]['discounted_return'].mean()

    # R-Max
    df = dfs[rmax_id]
    y.append(df.loc[df['task'] == 1]['discounted_return'].values / norm_1)
    labels.append(custom_label(agents[rmax_id], 1))

    # LR-Max
    df = dfs[lrmax_id]
    y.append(df.loc[df['task'] == 1]['discounted_return'].values / norm_1)
    labels.append(custom_label(agents[lrmax_id], 1))
    y.append(df.loc[df['task'] == 2]['discounted_return'].values / norm_2)
    labels.append(custom_label(agents[lrmax_id], 2))

    # MaxQInit
    df = dfs[maxqinit_id]
    y.append(df.loc[df['task'] == 11]['discounted_return'].values / norm_11)
    labels.append(custom_label(agents[maxqinit_id], 11))
    y.append(df.loc[df['task'] == 12]['discounted_return'].values / norm_12)
    labels.append(custom_label(agents[maxqinit_id], 12))

    y_lab = r'Relative discounted return'
    plot(path, pdf_name='custom_lifelong', agents=None, x=x, y=y, y_lo=None, y_up=None, labels=labels,
         x_label='Episode number', y_label=y_lab, open_plot=False, plot_title=False,
         plot_legend=True, legend_at_bottom=True, title_prefix='', plot_markers=False,
         ma=True, ma_width=200, latex_rendering=True, custom=True)


def lifelong_plot(
        agents,
        path,
        n_tasks,
        n_episodes,
        confidence,
        open_plot,
        plot_title,
        plot_legend=True,
        legend_at_bottom=False,
        episodes_moving_average=False,
        episodes_ma_width=10,
        tasks_moving_average=False,
        tasks_ma_width=10,
        latex_rendering=False,
        figure_title=None
):
    """
    Special plot routine for lifelong experiments.
    :param agents: (list)
    :param path: (str)
    :param n_tasks: (int)
    :param n_episodes: (int)
    :param confidence: (float)
    :param open_plot: (bool)
    :param plot_title: (bool)
    :param plot_legend: (int) takes several possible values:
        0: no legend
        1: only plot the legend for graphs displaying results w.r.t. episodes
        2: only plot the legend for graphs displaying results w.r.t. tasks
        3: legend for all
    :param legend_at_bottom: (bool)
    :param episodes_moving_average: (bool)
    :param episodes_ma_width: (int)
    :param tasks_moving_average: (bool)
    :param tasks_ma_width: (int)
    :param latex_rendering: (bool)
    :return: None
    """
    # TODO set those parameters:
    # norm_ag, which_norm_ag = (0, 0) for tight
    norm_ag = None
    which_norm_ag = 0

    dfs = []
    for agent in agents:
        agent_path = csv_path_from_agent(path, agent)
        dfs.append(pandas.read_csv(agent_path))

    averaged_lifelong_plot(dfs, agents, path, n_tasks, n_episodes, confidence, open_plot, plot_title,
                           plot_legend=3, legend_at_bottom=legend_at_bottom, norm_ag=norm_ag,
                           which_norm_ag=which_norm_ag,
                           episodes_moving_average=episodes_moving_average, episodes_ma_width=episodes_ma_width,
                           tasks_moving_average=tasks_moving_average, tasks_ma_width=tasks_ma_width,
                           latex_rendering=latex_rendering, figure_title=figure_title)

def compute_ma(x, w):
    df = pandas.DataFrame(x)
    ma_df = df.rolling(window=w, min_periods=1, center=True).mean()
    return np.array(ma_df[0])


def moving_average(w, x, y, y_lo=None, y_up=None):
    """
    Compute the moving average.
    :param w: (int) width
    :param x: (array-like)
    :param y: (array-like)
    :param y_lo: (array-like)
    :param y_up: (array-like)
    :return:
    """
    assert w > 1, 'Error: moving average width must be > 1: w = {}'.format(w)
    assert len(x) == len(y), 'Error: x and y vector should have the same length: len(x) = {}, len(y) = {}'.format(
        len(x), len(y))
    x_new = x
    y_new = compute_ma(y, w)
    y_lo_new = None if y_lo is None else compute_ma(y_lo, w)
    y_up_new = None if y_lo is None else compute_ma(y_up, w)
    return x_new, y_new, y_lo_new, y_up_new


def sub_sampling(w, x, y, y_lo=None, y_up=None):
    """
    Compute a sub-sampling of the input arrays.
    :param w: (int) width
    :param x: (array-like)
    :param y: (array-like)
    :param y_lo: (array-like)
    :param y_up: (array-like)
    :return:
    """
    assert w > 1, 'Error: moving average width must be > 1: w = {}'.format(w)
    assert len(x) == len(y), 'Error: x and y vector should have the same length: len(x) = {}, len(y) = {}'.format(
        len(x), len(y))

    n = len(x)
    w_2 = int(w / 2)
    x, y = np.array(x), np.array(y)
    x_new, y_new = [], []
    y_lo_new = None if y_lo is None else []
    y_up_new = None if y_up is None else []
    indexes = list(range(w_2, n, w))

    for i in indexes:
        x_new.append(np.mean(x[i - w_2: i + w_2 - 1]))
        y_new.append(np.mean(y[i - w_2: i + w_2 - 1]))
        if y_lo is not None:
            y_lo_new.append(np.mean(y_lo[i - w_2: i + w_2 - 1]))
        if y_up is not None:
            y_up_new.append(np.mean(y_up[i - w_2: i + w_2 - 1]))

    x_new = np.insert(x_new, 0, x[0])
    x_new = np.append(x_new, x[-1])

    y_new = np.insert(y_new, 0, np.mean(y[0:w_2]))
    y_new = np.append(y_new, np.mean(y[-w_2:]))

    if y_lo is not None:
        y_lo_new = np.insert(y_lo_new, 0, np.mean(y_lo[0:w_2]))
        y_lo_new = np.append(y_lo_new, np.mean(y_lo[-w_2:]))

    if y_up is not None:
        y_up_new = np.insert(y_up_new, 0, np.mean(y_up[0:w_2]))
        y_up_new = np.append(y_up_new, np.mean(y_up[-w_2:]))

    return x_new, y_new, y_lo_new, y_up_new


def plot(
        path,
        pdf_name,
        agents,
        x,
        y,
        y_lo,
        y_up,
        x_label,
        y_label,
        title_prefix,
        labels=None,
        x_cut=None,
        decreasing_x_axis=False,
        open_plot=True,
        title=None,
        plot_title=True,
        plot_markers=True,
        marke_every=None,
        plot_legend=True,
        legend_at_bottom=False,
        ma=False,
        ma_width=10,
        latex_rendering=False,
        custom=False,
        figure_title=None
):
    """
    Standard plotting routine.
    :param path: (str) experiment path
    :param pdf_name: (str)
    :param agents: (list) list of agents
    :param x: (list) x axis data
    :param y: (list) list of array-like containing the x data for each agent
    :param y_lo: (list) list of array-like containing the lower bound on the confidence interval of the y data
    :param y_up: (list) list of array-like containing the upper bound on the confidence interval of the y data
    :param x_label: (str)
    :param y_label: (str)
    :param title_prefix: (str)
    :param labels: (list) list of labels if agents is None
    :param x_cut: (int) cut the x_axis, does nothing if set to None
    :param decreasing_x_axis: (bool)
    :param open_plot: (bool)
    :param title: (str)
    :param plot_title: (bool)
    :param plot_markers: (bool)
    :param plot_legend: (bool)
    :param legend_at_bottom: (bool)
    :param ma: (bool) Moving average
    :param ma_width: (int)
    :param latex_rendering: (bool)
    :param custom: (bool)
    :return: None
    """
    # Font size and LaTeX rendering
    # matplotlib.rcParams["figure.figsize"] = [6.4, 5.5]  # default: [6.4, 4.8]  # TODO remove
    matplotlib.rcParams.update({'font.size': FONT_SIZE})  # default: 10
    if latex_rendering:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_aspect(5./ax.get_data_ratio())

    # Parse labels
    if agents is None:
        n_curves = len(labels)
    else:
        n_curves = len(agents)
        labels = []
        for i in range(n_curves):
            labels.append(_format_label(str(agents[i]), latex_rendering))
    # x-cut
    if x_cut is not None:
        x = x[:x_cut]
        for i in range(n_curves):
            y[i] = y[i][:x_cut]
            y_lo[i] = y_lo[i][:x_cut]
            y_up[i] = y_up[i][:x_cut]

    # Set markers and colors
    markers = ['o', 's', 'D', '^', '*', 'x', 'p', '+', 'v', '|']
    colors = [[shade / 255.0 for shade in rgb] for rgb in COLOR_LIST]
    colors = colors[COLOR_SHIFT:] + colors[:COLOR_SHIFT]
    ax.set_prop_cycle(cycler('color', colors))

    for i in range(n_curves):
        if ma:
            if y_lo is not None and y_up is not None:
                _x, y[i], y_lo[i], y_up[i] = moving_average(ma_width, x, y[i], y_lo[i], y_up[i])
            else:
                _x, y[i], _, _ = moving_average(ma_width, x, y[i], None, None)
        else:
            _x = x
        if y_lo is not None and y_up is not None:
            c_i = colors[i % len(colors)]
            plt.fill_between(_x, y_lo[i], y_up[i], alpha=0.25, facecolor=c_i, edgecolor=c_i)
        if plot_markers:
            plt.plot(_x, y[i], '-o', label=labels[i], marker=markers[i % len(markers)], markersize=3.0, markevery=marke_every)
        else:
            plt.plot(_x, y[i], label=labels[i])

    # x y labels
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylim(bottom=0)
    if decreasing_x_axis:
        plt.xlim(max(x), min(x))

    if custom:
        ax.yaxis.set_label_coords(-0.1, 0.1)
        #plt.figure(figsize=(20, 20))

    if plot_legend:
        if legend_at_bottom:
            # Shrink current axis's height by p% on the bottom
            p = 0.4
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * p, box.width, box.height * (1.0 - p)])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=False, shadow=False, ncol=2)
        else:
            # plt.legend(loc='lower right', fontsize=10, ncol=2)
            # plt.legend(loc='best', fontsize=10, ncol=2)
            print('1')

    plt.grid(True, linestyle='--')

    exp_dir_split_list = path.split("/")
    if figure_title is None:
        if 'results' in exp_dir_split_list:
            exp_name = exp_dir_split_list[exp_dir_split_list.index('results') + 1]
        else:
            exp_name = exp_dir_split_list[0]
    else:
        exp_name = figure_title
    # print(exp_name)
    if plot_title:
        plt_title = _format_title(title) if title is not None else _format_title(title_prefix + exp_name)
        plt.title(plt_title)

    # Save
    plot_file_name = os.path.join(path, pdf_name + '.pdf')
    plt.savefig(plot_file_name, format='pdf')

    # Open
    # if open_plot:
    #     open_prefix = 'gnome-' if sys.platform == 'linux' or sys.platform == 'linux2' else ''
    #     # os.system(open_prefix + 'open ' + plot_file_name)
    #     os.system(open_prefix + plot_file_name)

    # Clear and close
    plt.cla()
    plt.close()


def plot_color_bars(
        path,
        pdf_name,
        x,
        y,
        y_lo,
        y_up,
        cb_min,
        cb_max,
        cb_step,
        x_label,
        y_label,
        title_prefix,
        labels,
        cbar_label=None,
        x_cut=None,
        decreasing_x_axis=False,
        open_plot=False,
        title=None,
        plot_title=False,
        plot_markers=True,
        plot_legend=False,
        legend_at_bottom=False,
        ma=True,
        ma_width=10,
        latex_rendering=False
):
    """
    Standard plotting routine with color bars.
    :param path: (str) experiment path
    :param pdf_name: (str)
    :param x: (list) x axis data
    :param y: (list) list of array-like containing the x data for each agent
    :param y_lo: (list) list of array-like containing the lower bound on the confidence interval of the y data
    :param y_up: (list) list of array-like containing the upper bound on the confidence interval of the y data
    :param x_label: (str)
    :param y_label: (str)
    :param title_prefix: (str)
    :param labels: (list) list of labels if agents is None
    :param x_cut: (int) cut the x_axis, does nothing if set to None
    :param decreasing_x_axis: (bool)
    :param open_plot: (bool)
    :param title: (str)
    :param plot_title: (bool)
    :param plot_markers: (bool)
    :param plot_legend: (bool)
    :param legend_at_bottom: (bool)
    :param ma: (bool)
    :param ma_width: (int)
    :param latex_rendering: (bool)
    :return: None
    """
    # Font size and LaTeX rendering
    matplotlib.rcParams.update({'font.size': FONT_SIZE})  # default: 10
    if latex_rendering:
        rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Labels
    n_curves = len(labels)  # number of curves
    for i in range(len(labels)):
        labels[i] = _format_label(labels[i], latex_rendering)

    # x-cut
    if x_cut is not None:
        x = x[:x_cut]
        for i in range(n_curves):
            y[i] = y[i][:x_cut]
            y_lo[i] = y_lo[i][:x_cut]
            y_up[i] = y_up[i][:x_cut]

    # Markers and colors
    markers = ['o', 's', 'D', '^', '*', 'x', 'p', '+', 'v', '|']
    cb_parameters = np.array(range(cb_min, cb_max, cb_step))
    norm = matplotlib.colors.Normalize(vmin=np.min(cb_parameters), vmax=np.max(cb_parameters))
    c_m = matplotlib.cm.summer  # color map

    # create a ScalarMappable and initialize a data structure
    s_m = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
    s_m.set_array([])

    for i in range(n_curves):
        color_i = s_m.to_rgba(cb_parameters[i])
        if ma:
            if y_lo is not None and y_up is not None:
                _x, y[i], y_lo[i], y_up[i] = moving_average(ma_width, x, y[i], y_lo[i], y_up[i])
            else:
                _x, y[i], _, _ = moving_average(ma_width, x, y[i], None, None)
        else:
            _x = x

        # Interval plot
        if y_lo is not None and y_up is not None:
            plt.fill_between(_x, y_lo[i], y_up[i], alpha=0.25, facecolor=color_i, edgecolor=color_i)

        # Mean plot
        if plot_markers:
            plt.plot(_x, y[i], '-o', label=labels[i], marker=markers[i % len(markers)], color=color_i)
        else:
            plt.plot(_x, y[i], label=labels[i], color=color_i)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylim(bottom=0)
    if decreasing_x_axis:
        plt.xlim(max(x), min(x))

    # Legend
    if plot_legend:
        if legend_at_bottom:
            # Shrink current axis's height by p% on the bottom
            p = 0.4
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * p, box.width, box.height * (1.0 - p)])
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=2)
        else:
            plt.legend(loc='best')

    # Grid and color bar
    plt.grid(True, linestyle='--')
    cbar = plt.colorbar(s_m)
    if cbar_label is not None:
        cbar.set_label(cbar_label, rotation=270)

    exp_dir_split_list = path.split("/")
    if 'results' in exp_dir_split_list:
        exp_name = exp_dir_split_list[exp_dir_split_list.index('results') + 1]
    else:
        exp_name = exp_dir_split_list[0]
    if plot_title:
        plt_title = _format_title(title) if title is not None else _format_title(title_prefix + exp_name)
        plt.title(plt_title)

    # Save
    plot_file_name = os.path.join(path, pdf_name + '.pdf')
    plt.savefig(plot_file_name, format='pdf')

    # Open
    if open_plot:
        open_prefix = 'gnome-' if sys.platform == 'linux' or sys.platform == 'linux2' else ''
        os.system(open_prefix + 'open ' + plot_file_name)

    # Clear and close
    plt.cla()
    plt.close()


def _format_title(title):
    title = title.replace("_", " ")
    title = title.replace("-", " ")
    if len(title.split(" ")) > 1:
        return " ".join([w[0].upper() + w[1:] for w in title.strip().split(" ")])


def _format_label(label, latex_rendering):
    if latex_rendering:
        label = label.replace('Dmax=', r'$D_{\max} =$ ')
    return label
