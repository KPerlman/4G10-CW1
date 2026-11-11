import matplotlib.pyplot as plt
import cond_color
import numpy as np

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

def plot_pc_trajectories(
    Z_trajectories,
    title="Principal Component Trajectories",
    xlabel="PC1 Score",
    ylabel="PC2 Score"
):
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

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()

    return fig

def plot_dynamics_matrix(A):
    fig, ax = plt.subplots()
    limit = np.max(np.abs(A))
    image = ax.imshow(A, cmap="RdBu_r", vmin=-limit, vmax=limit)
    fig.colorbar(image, ax=ax, label="Predicted change per unit current PC coordinate")
    ax.set_title("Fitted Dynamics Matrix A")
    ax.set_xlabel("Current PC Coordinate")
    ax.set_ylabel("Predicted change in PC Coordinate")
    fig.tight_layout()

    return fig

def plot_premovement_comparison(
    movement_trajectories,
    premovement_trajectories
):
    fig = plot_pc_trajectories(
        movement_trajectories,
        title="Pre-movement and movement activity",
        xlabel="FR axis 1",
        ylabel="FR axis 2"
    )
    ax = fig.axes[0]

    # Fade movement markers because it was cluttered
    for collection in ax.collections:
        collection.set_alpha(0.25)

    # Reduce opacity of movement so premovement is more visible
    for line in ax.lines:
        line.set_alpha(0.25)

    end_x = premovement_trajectories[0, :, -1]
    end_y = premovement_trajectories[1, :, -1]

    colors = cond_color.get_colors(end_x, end_y, alt_colors=True)

    # for each condition
    # Select coordinates 0 and 1 from premovement_trajectories. plot them on ax, using colors[condition].
    for condition in range(premovement_trajectories.shape[1]):
        pc1 = premovement_trajectories[0, condition, :]
        pc2 = premovement_trajectories[1, condition, :]
        ax.plot(pc1, pc2, color=colors[condition])

    start_x = premovement_trajectories[0, :, 0]
    start_y = premovement_trajectories[1, :, 0]
    cond_color.plot_start(start_x, start_y, colors, markersize=80, ax=ax)
    cond_color.plot_end(end_x, end_y, colors, markersize=25, ax=ax)
    fig.tight_layout()
    return fig