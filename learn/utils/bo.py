import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def get_reward_euler(next_ob, action, pr=[0,1]):
    # Going to make the reward -c(x) where x is the attitude based cost
    assert isinstance(next_ob, np.ndarray)
    assert isinstance(action, np.ndarray)
    assert next_ob.ndim in (1, 2)

    was1d = next_ob.ndim == 1
    if was1d:
        next_ob = np.expand_dims(next_ob, 0)
        action = np.expand_dims(action, 0)
    assert next_ob.ndim == 2

    pitch = np.divide(next_ob[:, pr[0]], 180)
    roll = np.divide(next_ob[:, pr[1]], 180)
    cost_pr = np.power(pitch, 2) + np.power(roll, 2)
    cost_rates = np.power(next_ob[:, 3], 2) + np.power(next_ob[:, 4], 2) + np.power(next_ob[:, 5], 2)
    lambda_omega = .0001
    cost = cost_pr + lambda_omega * cost_rates
    return cost

def plot_cost_itr(logs, cfg):
    # TODO make a plot with the cost over time of iterations (with minimum)
    itr = np.arange(0, logs.data.n_evals)
    costs = logs.data.fx
    best = logs.data.opt_fx
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cum_min = np.minimum.accumulate(costs.squeeze()) #- .02
    ax.step(itr, costs.squeeze(), where='mid', label='Cost at Iteration')  # drawstyle="steps-post",
    ax.step(itr, cum_min, where='mid', label='Best Cost')  # drawstyle="steps-post", l
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Normalized Cumulative Attitude Cost")

    low, up = ax.get_ylim()
    l = mlines.Line2D([0, 15], [1.05*low, 1.05*low], color='red', label='Random Samples')
    ax.add_line(l)
    ax.legend()

    # ax.set_ylim([0, 5])
    fig.savefig("costs.pdf")
    ax.clear()
    return


def plot_parameters(logs, cfg, pid_s):
    # Returns a plot of the different PID parameters tested,
    # with the P and D plotted separately
    best = pid_s.transform(np.array(logs.data.opt_x).squeeze())
    samples = np.stack([pid_s.transform(s) for s in np.array(logs.data.x).T])  # np.array(logs.data.x).T
    Kp1 = samples[:, 0]
    Kp2 = samples[:, 3]
    Kd1 = samples[:, 2]
    Kd2 = samples[:, 5]
    Ki1 = samples[:, 1]
    Ki2 = samples[:, 4]


    fig2 = plt.figure()
    ax1 = fig2.add_subplot(1, 1, 1)
    ax1.scatter(Kp1, Kp2, label='sampled Kp')
    ax1.scatter(best[0], best[3], marker='*', s=130, label='Best Kp')
    ax1.set_ylim([0, cfg.policy.pid.params.max_values[0]])
    ax1.set_xlim([0, cfg.policy.pid.params.max_values[0]])
    ax1.set_xlabel("KP Pitch")
    ax1.set_ylabel("KP Roll")
    ax1.legend()
    fig2.savefig("KP.pdf")
    ax1.clear()

    fig3 = plt.figure()
    ax2 = fig3.add_subplot(1, 1, 1)
    ax2.scatter(Kd1, Kd2, label='sampled Kd')
    ax2.scatter(best[2], best[5], marker='*', s=130, label='Best Kd')
    ax2.set_ylim([0, cfg.policy.pid.params.max_values[2]])
    ax2.set_xlim([0, cfg.policy.pid.params.max_values[2]])
    ax2.set_xlabel("KD Pitch")
    ax2.set_ylabel("KD Roll")
    ax2.legend()
    fig3.savefig("KD.pdf")
    ax2.clear()

    fig4 = plt.figure()
    ax3 = fig4.add_subplot(1, 1, 1)
    ax3.scatter(Ki1, Ki2, label='sampled Ki')
    ax3.scatter(best[1], best[4], marker='*', s=130, label='Best Ki')
    ax3.set_ylim([0, cfg.policy.pid.params.max_values[1]])
    ax3.set_xlim([0, cfg.policy.pid.params.max_values[1]])
    ax3.set_xlabel("KI Pitch")
    ax3.set_ylabel("KI Roll")
    ax3.legend()
    fig4.savefig("KI.pdf")
    ax3.clear()

    return


from sklearn.preprocessing import MinMaxScaler


class PID_scalar():
    def __init__(self, policy_cfg):
        self.scalar_p = MinMaxScaler()
        self.scalar_i = MinMaxScaler()
        self.scalar_d = MinMaxScaler()

        self.scalar_p.fit([[policy_cfg.pid.params.min_values[0]], [policy_cfg.pid.params.max_values[0]]])
        self.scalar_i.fit([[policy_cfg.pid.params.min_values[1]], [policy_cfg.pid.params.max_values[1]]])
        self.scalar_d.fit([[policy_cfg.pid.params.min_values[2]], [policy_cfg.pid.params.max_values[2]]])

    def transform(self, PID):
        if len(PID) == 6:
            return np.concatenate((self.transform(PID[:3]), self.transform(PID[3:])))
        else:
            return np.squeeze([self.scalar_p.inverse_transform(PID[0]), self.scalar_i.inverse_transform(PID[1]),
                               self.scalar_d.inverse_transform(PID[2])])
