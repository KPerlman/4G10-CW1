import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_individual_psths, plot_population_mean_psth

# Import data
data = np.load("data/psths.npz")
X = data["X"]
times = data["times"]

# Exercise 1: Plot PSTH
neurons = [0, 20, 60, 100]
conditions = [0, 1, 2, 3, 4]

fig = plot_individual_psths(X, times, neurons, conditions)
fig.savefig("outputs/e1_individual_psths.pdf", bbox_inches="tight")
plt.show()

# Exercise 1: Plot population mean PSTH
population_mean = X.mean(axis=(0, 1))
fig = plot_population_mean_psth(X, times)
plt.savefig("outputs/e1_population_mean_psth.pdf", bbox_inches="tight")
plt.show()