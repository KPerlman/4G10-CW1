import matplotlib.pyplot as plt
import cond_color

def plot_individual_psths(X, times, neurons, conditions):
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

    for ax, neuron in zip(axes.flat, neurons):
        for condition in conditions:
            psth = X[neuron, condition, :]
            ax.plot(times, psth, label=f"Condition {condition}")

        ax.set_title(f"Neuron {neuron}: selected conditions")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Firing Rate (Hz)")
        ax.axvline(0, color="grey", linestyle="--")

    axes[0,0].legend()
    fig.tight_layout()

    return fig

def plot_population_mean_psth(X, times):
    fig, ax = plt.subplots()
    ax.plot(times, X.mean(axis=(0, 1)))
    ax.set_title("Population mean PSTH")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.axvline(0, color="grey", linestyle="--")
    fig.tight_layout()

    return fig

def plot_neuron_maxima(neuron_maxima):
    fig, ax = plt.subplots()
    ax.hist(neuron_maxima, bins=20)
    ax.set_title("Distribution of Neuron Maxima")
    ax.set_xlabel("Max Firing Rate (Hz)")
    ax.set_ylabel("Number of Neurons")
    fig.tight_layout()

    return fig

def plot_pc_trajectories(Z_trajectories):
    start_x = Z_trajectories[0, :, 0]
    start_y = Z_trajectories[1, :, 0]
    colors = cond_color.get_colors(start_x, start_y)
    
    fig, ax = plt.subplots()
    for condition in range(Z_trajectories.shape[1]):
        pc1 = Z_trajectories[0, condition, :]
        pc2 = Z_trajectories[1, condition, :]
        ax.plot(pc1, pc2, color=colors[condition])

    end_x = Z_trajectories[0, :, -1]
    end_y = Z_trajectories[1, :, -1]

    cond_color.plot_start(start_x, start_y, colors, markersize=80, ax=ax)
    cond_color.plot_end(end_x, end_y, colors, markersize=25, ax=ax)

    ax.set_title("Principal Component Trajectories")
    ax.set_xlabel("PC1 Score")
    ax.set_ylabel("PC2 Score")
    fig.tight_layout()

    return fig