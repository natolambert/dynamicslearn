# File for plotting utilities

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
import os

import plotly
import matplotlib.pyplot as plt

from .sim import gather_predictions

colors = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]

markers = [
    "cross-open-dot",
    "circle-open-dot",
    "x-open-dot",
    "triangle-up-open-dot",
    "y-down-open",
    "diamond-open-dot",
    "hourglass-open",
    "hash-open-dot",
    "star-open-dot",
    "square-open-dot",
]


def plot_results(logx=False, logy=False, save=False):
    import glob
    import torch
    SAC_example = '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/sweeps/2020-03-25/20-30-47/'
    # '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/sweeps/2020-03-25/18-46-58/'
    MPC_example = '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/sweeps/2020-03-25/20-30-57/'
    # '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/sweeps/2020-03-25/19-48-27/'
    MPC_example_lowN = '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/sweeps/2020-03-25/20-33-38/'
    MPC_example_lowNT = '/Users/nato/Documents/Berkeley/Research/Codebases/dynamics-learn/sweeps/2020-03-26/11-01-16/'

    sweep_dirs = [SAC_example, MPC_example, MPC_example_lowN,  MPC_example_lowNT]
    alg_names = ['SAC', 'MPC', 'MPC-Low N', 'MPC-Low N&T']
    labels = []
    rewards = []
    samples = []
    algs = []
    # outer loop is control type
    for dir, alg in zip(sweep_dirs, alg_names):
        sub_dirs = os.listdir(dir)
        # This sweeps across the reward functions eg
        for sub in sub_dirs:
            # this
            path = os.path.join(dir, sub)
            seeds = glob.glob(path + "/*")
            # This sweeps across the random seeds
            seed_r = []
            seed_samples = []
            for s in seeds:
                trials = glob.glob(s + '/trial_*')
                trials.sort(key=lambda x: os.path.getmtime(x))
                data = torch.load(trials[-1])
                seed_r.append(data['rewards'])
                seed_samples.append(data['steps'])
            rewards.append(seed_r)
            samples.append(seed_samples)
            algs.append(alg)
            labels.append(sub[sub.find("name=") + len("name="):sub.find(',')])

    import plotly
    import plotly.graph_objects as go

    unique_labels = np.unique(labels)
    traces = []
    fig = plotly.subplots.make_subplots(rows=len(unique_labels), cols=1,
                                        subplot_titles=(unique_labels),
                                        vertical_spacing=0.1,
                                        shared_xaxes=True, )  # go.Figure()
    for p, u in enumerate(unique_labels):
        idx = np.argwhere(np.array(labels) == u)
        for c, i in enumerate(idx.squeeze()):
            to_plot_rew = rewards[i]
            to_plot_samples = samples[i]
            alg = algs[i]
            lab = labels[i]

            min_len = min([len(r) for r in to_plot_rew])
            to_plot_rew = np.stack([t[:min_len] for t in to_plot_rew]).squeeze()
            to_plot_samples = np.stack([t[:min_len] for t in to_plot_samples]).squeeze()

            rew_mean = np.mean(np.stack(to_plot_rew).squeeze(), axis=0)
            rew_std = np.std(np.stack(to_plot_rew).squeeze(), axis=0)
            samp_mean = np.mean(np.stack(to_plot_samples).squeeze(), axis=0)
            samp_std = np.std(np.stack(to_plot_samples).squeeze(), axis=0)

            tr, xs, ys = generate_errorbar_traces(ys=to_plot_rew.tolist(), xs=[to_plot_samples[0].tolist()],
                                                  color=colors[c], name=alg)

            # fig.add_trace(go.Scatter(y=rew_mean, x=samp_mean, name=alg + lab, legendgroup=alg,
            #                          error_y=dict(
            #                              type='data',  # value of error bar given in data coordinates
            #                              array=rew_std,
            #                              visible=True),
            #                          error_x=dict(
            #                              type='data',  # value of error bar given in data coordinates
            #                              array=samp_std,
            #                              visible=True)
            #                          # line=dict(color=colors[i], width=2),  # mode='lines+markers',
            #                          # marker=dict(color=colors[i], symbol=markers[i], size=16)
            #                          ), row=p, col=1)
            for t in tr:
                t['legendgroup'] = alg
                if p > 0:
                    t['showlegend'] = False
                fig.add_trace(t, p + 1, 1)

    if logx:
        fig.update_xaxes(title_text="Env Step", type='log', row=3, col=1)
        fig.update_xaxes(title_text="Env Step", type='log', row=3, col=1)
        fig.update_xaxes(title_text="Env Step", type='log', row=3, col=1)
    else:
        fig.update_xaxes(title_text="Env Step", range=[0, 50000], row=3, col=1)
        fig.update_xaxes(range=[0, 50000], row=2, col=1)
        fig.update_xaxes(range=[0, 50000], row=1, col=1)

    if logy:
        fig.update_yaxes(title_text="Living Rew.", type='log', row=1, col=1)
        fig.update_yaxes(title_text="Rotation Rew.", type='log', row=2, col=1)
        fig.update_yaxes(title_text="Square Cost", type='log', row=3, col=1)
    else:
        fig.update_yaxes(title_text="Living Rew.", row=1, col=1)
        fig.update_yaxes(title_text="Rotation Rew.", row=2, col=1)
        fig.update_yaxes(title_text="Square Cost", range=[-50, 0], row=3, col=1)

    fig.update_layout(title=f"Sample Efficiency vs Reward",
                      font=dict(
                          family="Times New Roman, Times, serif",
                          size=24,
                          color="black"
                      ),
                      # legend_orientation="h",
                      # legend=dict(x=.05, y=0.5,
                      #             bgcolor='rgba(205, 223, 212, .4)',
                      #             bordercolor="Black",
                      #             ),
                      # xaxis_title='Timestep',
                      # yaxis_title='Angle (Degrees)',
                      plot_bgcolor='white',
                      width=1500,
                      height=800,
                      xaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      yaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      legend_orientation="h",
                      legend=dict(x=.7, y=0.1,
                                  bgcolor='rgba(205, 223, 212, .4)',
                                  bordercolor="Black",
                                  ),
                      )

    if save:
        fig.write_image(os.getcwd() + "/learning.png")
    else:
        fig.show()


def plot_dist(df, x, y, z):
    import plotly.express as px
    from scipy import stats

    fig = px.scatter_3d(df, x=x, y=y, z=z, opacity=0.7)  # , color='timesteps')
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),
        title=f"Training Data Distribution",
        # xaxis={'title': 'Trial Num'},
        # yaxis={'title': 'Cum. Reward'},
        font=dict(family='Times New Roman', size=18, color='#7f7f7f'),
        height=1200,
        width=1200,
        plot_bgcolor='white',
        scene_aspectmode='cube',
        scene=dict(
            xaxis=dict(nticks=8,  # range=[-100, 100],
                       showline=True,
                       showgrid=False,
                       showticklabels=True, ),
            yaxis=dict(nticks=8,  # range=[-50, 100],
                       showline=True,
                       showgrid=False,
                       showticklabels=True,
                       ),
            zaxis=dict(nticks=8,  # range=[-100, 100],
                       showline=True,
                       showgrid=False,
                       showticklabels=True,
                       ), ),
        # legend={'x': .83, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'}
    )
    fig.write_image(os.getcwd() + "/distribution.pdf")
    # fig.show()


def plot_eulers_action(df):
    raise NotImplementedError()


def plot_test_train(model, dataset, variances=False):
    """
    Takes a dynamics model and plots test vs train predictions on a dataset of the form (X,U,dX)
    - variances adds a highlight showing the variance of one step prediction estimates
    """
    if variances:
        predictions_means, predictions_vars = gather_predictions(
            model, dataset, variances=variances)
    else:
        predictions_means = gather_predictions(
            model, dataset, variances=variances)

    X = dataset[0]
    U = dataset[1]
    dX = dataset[2]

    dim = 0
    # New plot
    font = {'size': 11}

    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=True)
    matplotlib.rc('lines', linewidth=1.5)

    plt.tight_layout()

    # plot for test train compare
    fig = plt.figure()
    with sns.axes_style("whitegrid"):
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["axes.edgecolor"] = "0.15"
        plt.rcParams["axes.linewidth"] = 1.5
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        plt.subplots_adjust(bottom=.13, top=.93, left=.1, right=1 - .03, hspace=.28)

    # sort and plot data
    if not variances:
        # Gather test train splitted data
        lx = int(np.shape(dX)[0] * .8)
        data_train = zip(dX[:lx, dim], predictions_means[:lx, dim])
        data_train = sorted(data_train, key=lambda tup: tup[0])
        gt_sort_train, pred_sort_pll_train = zip(*data_train)

        data_test = zip(dX[lx:, dim], predictions_means[lx:, dim])
        data_test = sorted(data_test, key=lambda tup: tup[0])
        gt_sort_test, pred_sort_pll_test = zip(*data_test)

        # plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tick_params(axis='both', which='minor', labelsize=7)

        gt_train = np.linspace(0, 1, len(gt_sort_train))
        ax1.plot(gt_train, gt_sort_train, label='Ground Truth', color='k', linewidth=1.8)
        ax1.plot(gt_train, pred_sort_pll_train, '-', label='Probablistic Model Prediction',
                 markersize=.9, linewidth=.7, alpha=.8)  # , linestyle=':')
        ax1.set_title("Training Data Predictions")
        ax1.legend(prop={'size': 7})

        gt_test = np.linspace(0, 1, len(gt_sort_test))
        ax2.plot(gt_test, gt_sort_test, label='Ground Truth', color='k', linewidth=1.8)
        ax2.plot(gt_test, pred_sort_pll_test, '-', label='Bayesian Model Validation DataPrediction',
                 markersize=.9, linewidth=1.2, alpha=.8)  # , linestyle=':')
        ax2.set_title("Test Data Predictions")

    else:
        # Gather test train splitted data
        lx = int(np.shape(dX)[0] * .8)
        data_train = zip(dX[:lx, dim], predictions_means[:lx, dim], predictions_vars[:lx, dim])
        data_train = sorted(data_train, key=lambda tup: tup[0])
        gt_sort_train, pred_sort_pll_train, pred_vars_train = zip(*data_train)

        data_test = zip(dX[lx:, dim], predictions_means[lx:,
                                      dim], predictions_vars[lx:, dim])
        data_test = sorted(data_test, key=lambda tup: tup[0])
        gt_sort_test, pred_sort_pll_test, pred_vars_test = zip(*data_test)

        # plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tick_params(axis='both', which='minor', labelsize=7)

        gt_train = np.linspace(0, 1, len(gt_sort_train))
        ax1.plot(gt_train, np.array(pred_sort_pll_train) + np.array(pred_vars_train),
                 '-', color='r', label='Variance of Predictions', linewidth=.4, alpha=.4)
        ax1.plot(gt_train, np.array(pred_sort_pll_train) - np.array(pred_vars_train), '-', color='r', linewidth=.4,
                 alpha=.4)
        ax1.plot(gt_train, gt_sort_train, label='Ground Truth',
                 color='k', linewidth=1.8)
        ax1.plot(gt_train, pred_sort_pll_train, '-', label='Probablistic Model Prediction',
                 markersize=.9, linewidth=.7, alpha=.8)  # , linestyle=':')
        ax1.set_title("Training Data Predictions")
        ax1.legend(prop={'size': 7})

        gt_test = np.linspace(0, 1, len(gt_sort_test))
        ax2.plot(gt_test, np.array(pred_sort_pll_test) + np.array(pred_vars_test),
                 '-', color='r', label='Variance of Predictions', linewidth=.4, alpha=.4)
        ax2.plot(gt_test, np.array(pred_sort_pll_test) - np.array(pred_vars_test), '-', color='r', linewidth=.4,
                 alpha=.4)
        ax2.plot(gt_test, gt_sort_test, label='Ground Truth',
                 color='k', linewidth=1.8)
        ax2.plot(gt_test, pred_sort_pll_test, '-', label='Bayesian Model Validation DataPrediction',
                 markersize=.9, linewidth=1.2, alpha=.8)  # , linestyle=':')
        ax2.set_title("Test Data Predictions")

    mse = np.mean((np.array(pred_sort_pll_test) - gt_sort_test) ** 2)
    print(f"Mean sq error across test set: {mse}")

    fontProperties = {'family': 'Times New Roman'}

    # a = plt.gca()
    # print(a)
    # a.set_xticklabels(a.get_xticks(), fontProperties)
    # a.set_yticklabels(a.get_yticks(), fontProperties)

    ax1.grid(b=True, which='major', color='k',
             linestyle='-', linewidth=0, alpha=.75)
    ax1.grid(b=True, which='minor', color='b',
             linestyle='--', linewidth=0, alpha=.5)
    ax1.set_xticks([])
    ax2.grid(b=True, which='major', color='k',
             linestyle='-', linewidth=0, alpha=.75)
    ax2.grid(b=True, which='minor', color='b',
             linestyle='--', linewidth=0, alpha=.5)

    fig.text(.02, .75, 'One Step Prediction, Pitch (deg)',
             rotation=90, family='Times New Roman')
    # fig.text(.404, .04, 'Sorted Datapoints, Normalized', family='Times New Roman')
    ax2.set_xlabel('Sorted Datapoints, Normalized')

    for ax in [ax1, ax2]:
        # if ax == ax1:
        #     loc = matplotlib.ticker.MultipleLocator(base=int(lx/10))
        # else:
        #     loc = matplotlib.ticker.MultipleLocator(
        #         base=int((np.shape(dX)[0]-lx)/10))
        ax.set_ylim([-9.0, 9.0])
        ax.set_xlim([0, 1])

    fig.set_size_inches(5, 3.5)

    # plt.savefig('psoter', edgecolor='black', dpi=100, transparent=True)

    plt.savefig('testrain.pdf', format='pdf', dpi=300)

    # plt.show()
    return mse


def plot_rewards_over_trials(rewards, env_name, save=False):
    import plotly.graph_objects as go
    data = []
    traces = []
    colors = plt.get_cmap('tab10').colors

    i = 0
    cs_str = 'rgb' + str(colors[i])
    ys = np.stack(rewards)
    data.append(ys)
    err_traces, xs, ys = generate_errorbar_traces(np.asarray(data[i]), color=cs_str, name=f"simulation")
    for t in err_traces:
        traces.append(t)

    layout = dict(title=f"Learning Curve Reward vs Number of Trials (Env: {env_name})",
                  xaxis={'title': 'Trial Num'},
                  yaxis={'title': 'Cum. Reward'},
                  plot_bgcolor='white',
                  showlegend=False,
                  font=dict(family='Times New Roman', size=30, color='#7f7f7f'),
                  height=1000,
                  width=1500,
                  legend={'x': .83, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'})

    fig = {
        'data': traces,
        'layout': layout
    }

    fig = go.Figure(fig)

    if save:
        fig.write_image(os.getcwd() + "/learning.png")
    else:
        fig.show()

    return fig


def hv_characterization():
    d1 = '/Users/nato/Documents/Berkeley/Research/Ionocraft/data_setup/data/DACinHVout.csv'
    d2 = '/Users/nato/Documents/Berkeley/Research/Ionocraft/data_setup/data/IVstepping.csv'
    d3 = '/Users/nato/Documents/Berkeley/Research/Ionocraft/data_setup/data/stepresponse.csv'
    d4 = '/Users/nato/Documents/Berkeley/Research/Ionocraft/data_setup/data/weirdHVrail.csv'

    import plotly.io as pio
    import plotly.graph_objects as go

    fig = plotly.subplots.make_subplots(rows=3, cols=1,
                                        subplot_titles=(
                                            "100Hz Oscillation", "IV Step Response", "Voltage Step Response"),
                                        vertical_spacing=.25)  # go.Figure()

    fig.update_layout(title='HV Characterization',
                      plot_bgcolor='white',
                      xaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      yaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      # font=dict(family='Times New Roman', size=30, color='#7f7f7f'),
                      height=1000,
                      width=1500,
                      # legend={'x': .83, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'}
                      )
    with open(d1) as csvfile:
        # laod data
        df = pd.read_csv(csvfile, sep=",", header=10)

        dT = df.values[1:, 0] - df.values[:-1, 0]
        dT = np.mean(dT)
        time = (df.values[:-1, 0] - min(df.values[:-1, 0])) * 10 ** 3

        DAC = df.values[:-1, 1] * 100
        HV = df.values[:-1, 2]

        fig.add_trace(go.Scatter(x=time, y=DAC, name='DACx100',
                                 line=dict(color='firebrick', width=4), legendgroup='DAC'), row=1, col=1)
        fig.add_trace(go.Scatter(x=time, y=HV, name='HV',
                                 line=dict(color='royalblue', width=4), legendgroup='HV'), row=1, col=1)

        fig.update_xaxes(title_text='Time (ms)', row=1, col=1)
        fig.update_yaxes(title_text='Volts', row=1, col=1)

    with open(d2) as csvfile:
        # laod data
        df = pd.read_csv(csvfile, sep=",", header=10)

        dT = df.values[1:, 0] - df.values[:-1, 0]
        dT = np.mean(dT)
        time = (df.values[:-1, 0] - min(df.values[:-1, 0]))

        DAC = df.values[:-1, 1] * 100
        HV = df.values[:-1, 2]

        fig.add_trace(go.Scatter(x=time, y=DAC, name='DACx100',
                                 line=dict(color='firebrick', width=4), legendgroup='DAC', showlegend=False), row=2,
                      col=1)
        fig.add_trace(go.Scatter(x=time, y=HV, name='HV',
                                 line=dict(color='royalblue', width=4), legendgroup='HV', showlegend=False), row=2,
                      col=1)

        fig.update_xaxes(title_text='Time (s)', row=2, col=1)
        fig.update_yaxes(title_text='Volts', row=2, col=1)

    with open(d3) as csvfile:
        # laod data
        df = pd.read_csv(csvfile, sep=",", header=10)

        cutoff = 3000
        end_p = -2000
        dT = df.values[cutoff + 1:end_p, 0] - df.values[cutoff:-1 + end_p, 0]
        dT = np.mean(dT)
        time = (df.values[cutoff:-1 + end_p, 0] - min(df.values[cutoff:-1 + end_p, 0])) * 100 ** 3

        DAC = df.values[cutoff:-1 + end_p, 1] * 100
        HV = df.values[cutoff:-1 + end_p, 2]

        fig.add_trace(go.Scatter(x=time, y=DAC, name='DACx100',
                                 line=dict(color='firebrick', width=4), legendgroup='DAC', showlegend=False), row=3,
                      col=1)
        fig.add_trace(go.Scatter(x=time, y=HV, name='HV',
                                 line=dict(color='royalblue', width=4), legendgroup='HV', showlegend=False), row=3,
                      col=1)

        fig.update_xaxes(title_text='Time (ms)', row=3, col=1)
        fig.update_yaxes(title_text='Volts', row=3, col=1)

        ninety_rise = max(HV * .9)
        first = (HV > (max(HV) * .9)).nonzero()[0][0]
        t_90 = time[first]

        fig.add_trace(go.Scatter(
            x=[t_90 * .8],
            y=[ninety_rise * 1.2],
            text=[f"90 percent rise time = {round(t_90, 2)}, V = {round(ninety_rise, 2)}"],
            mode="text",
            showlegend=False,
        ),
            row=3, col=1
        )

        fig.add_shape(
            # Line Horizontal
            go.layout.Shape(
                type="line",
                x0=time[0],
                y0=ninety_rise,
                x1=time[-1],
                y1=ninety_rise,
                line=dict(
                    color="LightSeaGreen",
                    width=1,
                    # dash="dashdot",
                )
            ),
            row=3, col=1, )
        fig.add_shape(
            # Line Vertical
            go.layout.Shape(
                type="line",
                x0=t_90,
                y0=0,
                x1=t_90,
                y1=350,
                line=dict(
                    color="RoyalBlue",
                    width=1
                )
            ),
            row=3, col=1, )

    # fig.write_image(os.getcwd()+"/fig1.png")
    fig.write_image(os.getcwd() + "/fig1.pdf")
    pio.show(fig)


def generate_errorbar_traces(ys, xs=None, percentiles='66+95', color=None, name=None):
    if xs is None:
        xs = [list(range(len(y))) for y in ys]

    minX = min([len(x) for x in xs])

    if "#" in color:
        color = 'rgb' + str(tuple(int(color[i + 1:i + 3], 16) for i in (0, 2, 4)))

    xs = [x[:minX] for x in xs]
    ys = [y[:minX] for y in ys]

    assert np.all([(len(y) == len(ys[0])) for y in ys]), \
        'Y should be the same size for all traces'

    assert np.all([(x == xs[0]) for x in xs]), \
        'X should be the same for all traces'

    y = np.array(ys)

    def median_percentile(data, des_percentiles='66+95'):
        median = np.nanmedian(data, axis=0)
        out = np.array(list(map(int, des_percentiles.split("+"))))
        for i in range(out.size):
            assert 0 <= out[i] <= 100, 'Percentile must be >0 <100; instead is %f' % out[i]
        list_percentiles = np.empty((2 * out.size,), dtype=out.dtype)
        list_percentiles[0::2] = out  # Compute the percentile
        list_percentiles[1::2] = 100 - out  # Compute also the mirror percentile
        percentiles = np.nanpercentile(data, list_percentiles, axis=0)
        return [median, percentiles]

    out = median_percentile(y, des_percentiles=percentiles)
    ymed = out[0]

    err_traces = [
        dict(x=xs[0], y=ymed.tolist(), mode='lines', name=name, type='scatter', legendgroup=f"group-{name}",
             line=dict(color=color, width=4))]

    intensity = .3
    '''
    interval = scipy.stats.norm.interval(percentile/100, loc=y, scale=np.sqrt(variance))
    interval = np.nan_to_num(interval)  # Fix stupid case of norm.interval(0) returning nan
    '''

    for i, p_str in enumerate(percentiles.split("+")):
        p = int(p_str)
        high = out[1][2 * i, :]
        low = out[1][2 * i + 1, :]

        err_traces.append(dict(
            x=xs[0] + xs[0][::-1],
            type='scatter',
            y=(high).tolist() + (low).tolist()[::-1],
            fill='toself',
            fillcolor=(color[:-1] + str(f", {intensity})")).replace('rgb', 'rgba')
            if color is not None else None,
            line=dict(color='rgba(0,0,0,0)'),  # transparent'),
            # legendgroup=f"group-{name}",
            showlegend=False,
            name=name + str(f"_std{p}") if name is not None else None,
        ), )
        intensity -= .1

    return err_traces, xs, ys


def plot_rollout(states, actions, pry=[1, 2, 0], save=False, loc=None):
    import plotly.graph_objects as go
    import numpy as np
    import plotly
    ar = np.stack(states)
    l = np.shape(ar)[0]
    xs = np.arange(l)

    yaw = np.degrees(ar[:, pry[2]])
    pitch = np.degrees(ar[:, pry[0]])
    roll = np.degrees(ar[:, pry[1]])

    actions = np.stack(actions)

    fig = plotly.subplots.make_subplots(rows=2, cols=1,
                                        subplot_titles=("Euler Angles", "Actions"),
                                        vertical_spacing=.15)  # go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=yaw, name='Yaw',
                             line=dict(color='firebrick', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=pitch, name='Pitch',
                             line=dict(color='royalblue', width=4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=roll, name='Roll',
                             line=dict(color='green', width=4)), row=1, col=1)

    fig.add_trace(go.Scatter(x=xs, y=actions[:, 0], name='M1',
                             line=dict(color='firebrick', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 1], name='M2',
                             line=dict(color='royalblue', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 2], name='M3',
                             line=dict(color='green', width=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=actions[:, 3], name='M4',
                             line=dict(color='orange', width=4)), row=2, col=1)

    fig.update_layout(title='Euler Angles from MPC Rollout',
                      xaxis_title='Timestep',
                      yaxis_title='Angle (Degrees)',
                      plot_bgcolor='white',
                      xaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      yaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True, ),
                      )
    if save:
        fig.write_image(os.getcwd() + loc + "_rollout.png")
    else:
        fig.show()
