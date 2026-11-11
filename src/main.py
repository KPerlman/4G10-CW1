import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_individual_psths, plot_population_mean_psth, plot_neuron_maxima, plot_pc_trajectories, plot_dynamics_matrix
from dynamics import estimate_A, construct_H

# Load data
data = np.load("data/psths.npz")
X = data["X"]
times = data["times"]

############
# Exercise 1
############
# Plot PSTH
neurons = [0, 20, 60, 100]
conditions = [0, 1, 2, 3, 4]

fig = plot_individual_psths(X, times, neurons, conditions)
fig.savefig("outputs/e1_individual_psths.pdf", bbox_inches="tight")
# plt.show()

# Plot population mean PSTH
population_mean = X.mean(axis=(0, 1))
fig = plot_population_mean_psth(X, times)
fig.savefig("outputs/e1_population_mean_psth.pdf", bbox_inches="tight")
# plt.show()

############
# Exercise 2
############
neuron_maxima = X.max(axis=(1, 2))

# Plot maximas distribution
fig = plot_neuron_maxima(neuron_maxima)
fig.savefig("outputs/e2_neuron_maxima.pdf", bbox_inches="tight")
# plt.show()

# Normalize each neurons activities
neuron_maxima_broadcasted = X.max(axis=(1, 2), keepdims=True)
neuron_minima_broadcasted = X.min(axis=(1, 2), keepdims=True)
X_normalized = (X - neuron_minima_broadcasted) / (neuron_maxima_broadcasted - neuron_minima_broadcasted + 5)

# Mean centering
condition_mean = X_normalized.mean(axis=1, keepdims=True)
X_centered = X_normalized - condition_mean

movement_mask = (times >= -150) & (times <= 300)
X_movement = X_centered[:, :, movement_mask]
movement_times = times[movement_mask]
X_matrix = X_movement.reshape(182, -1)

# Compute covariance matrix
n_samples = X_matrix.shape[1]
covariance = (X_matrix @ X_matrix.T) / n_samples

# Get eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
V_M = eigenvectors[:, :12]
# Project data onto first 12 PCs
Z = V_M.T @ X_matrix

retained_variance = eigenvalues[:12].sum() / eigenvalues.sum()

############
# Exercise 3
############

Z_trajectories = Z.reshape(12, 108, 46)
pc1 = Z_trajectories[0, 0, :]
pc2 = Z_trajectories[1, 0, :]

fig = plot_pc_trajectories(Z_trajectories)
fig.savefig("outputs/e3_pc_trajectories.pdf", bbox_inches="tight")
# plt.show()

############
# Exercise 4
############


# E4b

H_small = construct_H(3) # 3 template matrices of size 3 x 3
beta_small = np.array([1, 2, 3])
A_small = np.tensordot(beta_small, H_small, axes=1)
print(A_small)

A_expected = np.array([
    [ 0,  1,  2],
    [-1,  0,  3],
    [-2, -3,  0]
])

print("Construction correct:", np.allclose(A_small, A_expected))
print("Antisymmetric:", np.allclose(A_small.T, -A_small))

# E4abcd

A_hat = estimate_A(Z_trajectories)

fig = plot_dynamics_matrix(A_hat)
fig.savefig("outputs/e4_dynamics_matrix.pdf", bbox_inches="tight")

# E4e

test_data = np.load("data/test.npz")
A_estimated_test = estimate_A(test_data["Z_test"])

max_error = np.max(np.abs(A_estimated_test - test_data["A_test"]))
print("Maximum test error:", max_error)

plt.show()