# File for plotting utilities

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

import plotly
import matplotlib.pyplot as plt


def plot_euler_preds(model, dataset):
    """
    returns a 3x1 plot of the Euler angle predictions for a given model and dataset
    """

    predictions_1 = gather_predictions(model, dataset)

    X = dataset[0]
    U = dataset[1]
    dX = dataset[2]

    dim = 3

    shift = 0
    # lx = int(n*.99)
    # Grab correction dimension data # for training :int(.8*n)

    if delta:
        ground_dim_1 = dX[:, 3]
        ground_dim_2 = dX[:, 4]
        ground_dim_3 = dX[:, 5]

    pred_dim_1 = predictions_1[:, 3]  # 3]
    pred_dim_2 = predictions_1[:, 4]  # 4]
    pred_dim_3 = predictions_1[:, 5]  # 5]
    global_dim_1 = X[:, 0 + shift + dim]  # 3
    global_dim_2 = X[:, 1 + shift + dim]  # 4
    global_dim_3 = X[:, 2 + shift + dim]  # 5

    # Sort with respect to ground truth
    # data = zip(ground_dim,pred_dim_1, ground_dim_2, ground_dim_3)
    # data = sorted(data, key=lambda tup: tup[0])
    # ground_dim_sort, pred_dim_sort_1, ground_dim_sort_2, ground_dim_sort_3 = zip(*data)

    # sorts all three dimenions for YPR
    data = zip(ground_dim_1, pred_dim_1, global_dim_1)
    data = sorted(data, key=lambda tup: tup[0])
    ground_dim_sort_1, pred_dim_sort_1, global_dim_sort_1 = zip(*data)

    data = zip(ground_dim_2, pred_dim_2, global_dim_2)
    data = sorted(data, key=lambda tup: tup[0])
    ground_dim_sort_2, pred_dim_sort_2, global_dim_sort_2 = zip(*data)

    data = zip(ground_dim_3, pred_dim_3, global_dim_3)
    data = sorted(data, key=lambda tup: tup[0])
    ground_dim_sort_3, pred_dim_sort_3, global_dim_sort_3 = zip(*data)

    font = {'size': 18}

    matplotlib.rc('font', **font)
    matplotlib.rc('lines', linewidth=2.5)

    # plt.tight_layout()

    with sns.axes_style("darkgrid"):
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313)

    my_dpi = 300
    plt.figure(figsize=(1200 / my_dpi, 1200 / my_dpi), dpi=my_dpi)
    ax1.axhline(0, linestyle=':', color='r', linewidth=1)
    ax1.plot(ground_dim_sort_1, label='Ground Truth', color='k', linewidth=1.8)
    ax1.plot(pred_dim_sort_1, ':', label='Model Prediction',
             markersize=.9, linewidth=.8)  # , linestyle=':')
    # ax1.set_xlabel('Sorted Datapoints')
    ax1.set_ylabel('Pitch Step (Deg.)')
    # ax1.set_ylim([-5,5])
    # ax1.set_yticks(np.arange(-5,5.01,2.5))

    # ax1.legend()
    # plt.show()

    # plt.title('One Step Dim+1')
    ax2.axhline(0, linestyle=':', color='r', linewidth=1)
    ax2.plot(ground_dim_sort_2, label='Ground Truth', color='k', linewidth=1.8)
    ax2.plot(pred_dim_sort_2, ':', label='Model Prediction',
             markersize=.9, linewidth=.8)  # , linestyle=':')

    # ax2.set_xlabel('Sorted Datapoints')
    ax2.set_ylabel('Roll Step (Deg.)')
    # ax2.set_ylim([-5,5])
    # ax2.set_yticks(np.arange(-5,5.01,2.5))
    # ax2.set_yticklabels(["-5", "-2.5", "0", "2.5", "5"])

    # ax2.legend()
    # plt.show()

    # plt.title('One Step Dim+2')
    ax3.axhline(0, linestyle=':', color='r', linewidth=1)
    ax3.plot(ground_dim_sort_3, label='Ground Truth', color='k', linewidth=1.8)
    ax3.plot(pred_dim_sort_3, ':', label='Model Prediction',
             markersize=.9, linewidth=.8)  # , linestyle=':')

    ax3.set_xlabel('Sorted Datapoints')
    ax3.set_ylabel('Yaw Step (Deg.)')
    ax3.set_ylim([-5, 5])
    ax3.set_yticks(np.arange(-5, 5.01, 2.5))
    leg3 = ax3.legend(loc=8, ncol=2)
    for line in leg3.get_lines():
        line.set_linewidth(2.5)
    plt.show()


def plot_test_train(model, dataset, variances=True):
    """
    Takes a dynamics model and plots test vs train predictions on a dataset of the form (X,U,dX)
    - variances adds a highlight showing the variance of one step prediction estimates
    """
    '''
    Some Models:
    model_pll = '_models/temp/2018-12-14--10-47-41.7_plot_pll_stack3_.pth'
    model_mse = '_models/temp/2018-12-14--10-51-10.9_plot_mse_stack3_.pth'
    model_pll_ens = '_models/temp/2018-12-14--10-53-42.9_plot_pll_ensemble_stack3_.pth'
    model_pll_ens_10 = '_models/temp/2018-12-14--11-49-21.6_plot_pll_ens_10_stack3_.pth'
    model_mse_ens = '_models/temp/2018-12-14--10-52-40.4_plot_mse_ensemble_stack3_.pth'

    25Hz models for with variance
    new ensemble: '_models/temp/2019-02-23--16-10-00.4_plot_temp__stack3_'
    new single: '_models/temp/2019-02-23--17-03-22.0_plot_temp_single_stack3_'
    '''
    # for crazyflie plots
    model_pll_ens_10 = '_models/temp/2018-12-14--11-49-21.6_plot_pll_ens_10_stack3_.pth'
    model_testing = '_models/temp/2019-02-25--09-51-49.1_temp_single_debug_stack3_.pth'

    # below for iono
    model_testing = "_models/temp/2019-05-02--09-43-01.5_temp_stack3_.pth"
    if variances:
        predictions_means, predictions_vars = gather_predictions(
            model_testing, dataset, variances=variances)
    else:
        predictions_means = gather_predictions(
            model_testing, dataset, variances=variances)

    X = dataset[0]
    U = dataset[1]
    dX = dataset[2]

    dim = 3
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

        print(np.shape(pred_sort_pll_train))
        print(np.shape(pred_vars_train))
        # plt.tick_params(axis='both', which='major', labelsize=10)
        plt.tick_params(axis='both', which='minor', labelsize=7)

        gt_train = np.linspace(0, 1, len(gt_sort_train))
        ax1.plot(gt_train, gt_sort_train, label='Ground Truth',
                 color='k', linewidth=1.8)
        ax1.plot(gt_train, pred_sort_pll_train, '-', label='Probablistic Model Prediction',
                 markersize=.9, linewidth=.7, alpha=.8)  # , linestyle=':')
        ax1.plot(gt_train, np.array(pred_sort_pll_train) + np.array(pred_vars_train),
                 '-', color='r', label='Variance of Predictions', linewidth=.4, alpha=.4)
        ax1.plot(gt_train, np.array(pred_sort_pll_train) - np.array(pred_vars_train), '-', color='r', linewidth=.4,
                 alpha=.4)
        ax1.set_title("Training Data Predictions")
        ax1.legend(prop={'size': 7})

        gt_test = np.linspace(0, 1, len(gt_sort_test))
        ax2.plot(gt_test, gt_sort_test, label='Ground Truth',
                 color='k', linewidth=1.8)
        ax2.plot(gt_test, pred_sort_pll_test, '-', label='Bayesian Model Validation DataPrediction',
                 markersize=.9, linewidth=1.2, alpha=.8)  # , linestyle=':')
        ax2.plot(gt_test, np.array(pred_sort_pll_test) + np.array(pred_vars_test),
                 '-', color='r', label='Variance of Predictions', linewidth=.4, alpha=.4)
        ax2.plot(gt_test, np.array(pred_sort_pll_test) - np.array(pred_vars_test), '-', color='r', linewidth=.4,
                 alpha=.4)
        ax2.set_title("Test Data Predictions")

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
        ax.set_ylim([-6.0, 6.0])
        ax.set_xlim([0, 1])

    fig.set_size_inches(5, 3.5)

    # plt.savefig('psoter', edgecolor='black', dpi=100, transparent=True)

    plt.savefig('testrain.pdf', format='pdf', dpi=300)

    # plt.show()


def plot_rewards_over_trials(rewards, env_name):
    data = []
    traces = []
    colors = plt.get_cmap('tab10').colors

    # for i, (log_dir, logs) in enumerate(all_logs.items()):
    #     string = log_dir
    i = 0
    cs_str = 'rgb' + str(colors[i])
    #     if i == 0: env_name = logs[0]['env_name']
    #     # new_data = [np.asarray(log['rewards']) for log in logs]
    #     # full_len = max([len(d) for d in new_data])
    #     # for vec in new_data:
    #     #     if len(vec) < full_len:
    #     #         np.concatenate
    # ys = np.stack([np.asarray(log['rewards']) for r in rewards])
    ys = np.stack(rewards)
    data.append(ys)
    err_traces, xs, ys = generate_errorbar_traces(np.asarray(data[i]), color=cs_str, name=f"simulation")
    for t in err_traces:
        traces.append(t)

    layout = dict(title=f"Learning Curve Reward vs Number of Trials (Env: {env_name})",
                  xaxis={'title': 'Trial Num'},
                  yaxis={'title': 'Cum. Reward'},
                  font=dict(family='Times New Roman', size=30, color='#7f7f7f'),
                  height=1000,
                  width=1500,
                  legend={'x': .83, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'})

    fig = {
        'data': traces,
        'layout': layout
    }

    import plotly.io as pio
    pio.show(fig)


def hv_characterization():
    d1 = '/Users/nato/Documents/Berkeley/Research/Ionocraft/Characterization/data/DACinHVout.csv'
    d2 = '/Users/nato/Documents/Berkeley/Research/Ionocraft/Characterization/data/IVstepping.csv'
    d3 = '/Users/nato/Documents/Berkeley/Research/Ionocraft/Characterization/data/stepresponse.csv'
    d4 = '/Users/nato/Documents/Berkeley/Research/Ionocraft/Characterization/data/weirdHVrail.csv'

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

    xs = [x[:minX] for x in xs]
    ys = [y[:minX] for y in ys]

    assert all([(len(y) == len(ys[0])) for y in ys]), \
        'Y should be the same size for all traces'

    assert all([(x == xs[0]) for x in xs]), \
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
            x=xs[0] + xs[0][::-1], type='scatter',
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
