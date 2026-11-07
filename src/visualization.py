import matplotlib.pyplot as plt

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

