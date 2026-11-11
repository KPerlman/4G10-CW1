import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_individual_psths, plot_population_mean_psth, plot_neuron_maxima, plot_pc_trajectories, plot_dynamics_matrix, plot_premovement_comparison
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

# plt.show()

############
# Exercise 5
############

# a)
rotation_eigenvals, rotation_eigenvecs = np.linalg.eig(A_hat)

fastest_index = np.argmax(rotation_eigenvals.imag)
fastest_vect = rotation_eigenvecs[:, fastest_index]

# b)
real_axis = fastest_vect.real
imag_axis = fastest_vect.imag

# normalize the real and imaginary axes to unit length
real_axis = fastest_vect.real / np.linalg.norm(fastest_vect.real)
imag_axis = fastest_vect.imag / np.linalg.norm(fastest_vect.imag)

# Projection matrix of fastest rotation plane
P_FR = np.vstack((real_axis, imag_axis))

# c)
# Project neural trajectories onto fastest rotation plane to get 2D coords
FR_coords = P_FR @ Z # 2 coords for each of 4968 snapshots
FR_trajectories = FR_coords.reshape(2, 108, 46)

# Plotting onto requested -150 to 200 ms window
FR_plot_mask = movement_times <= 200
FR_plot_trajectories = FR_trajectories[:, :, FR_plot_mask]

fig = plot_pc_trajectories(
    FR_plot_trajectories,
    title="Fastest-rotation plane",
    xlabel="FR axis 1",
    ylabel="FR axis 2"
)
fig.savefig("outputs/e5_fastest_rotation.pdf", bbox_inches="tight")

# d)
positive_idxs = np.flatnonzero(rotation_eigenvals.imag > 0) # indices of eigenvalues with positive imaginary parts

# sort speeds
pos_speeds = rotation_eigenvals.imag[positive_idxs]
speed_order = np.argsort(pos_speeds)[::-1] # sort in descending order
plane_idxs = positive_idxs[speed_order] # indices of eigenvals w positive imaginary parts sorted by speed

for rank, index in enumerate(plane_idxs[1:3], start=2):
    vector = rotation_eigenvecs[:, index]
    # normalize real and imaginary axes to unit length
    real_axis = vector.real / np.linalg.norm(vector.real)
    imag_axis = vector.imag / np.linalg.norm(vector.imag)
    # projection matrix of rotation plane
    P = np.vstack((real_axis, imag_axis))
    # project neural trajectories onto rotation plane to get 2D coords
    coords = P @ Z
    trajectories = coords.reshape(2, 108, 46)
    # plot onto window
    plot_mask = movement_times <= 200
    plot_trajectories = trajectories[:, :, plot_mask]
    # save
    fig = plot_pc_trajectories(
        plot_trajectories,
        title=f"Rotational plane {rank}",
        xlabel=f"Plane {rank} axis 1",
        ylabel=f"Plane {rank} axis 2"
    )
    fig.savefig(
        f"outputs/e5_rotation_plane_{rank}.pdf",
        bbox_inches="tight"
    )

############
# Exercise 6
############

# Isolate premovement data
premovement_mask = (times >= -800) & (times <= -150)
X_premovement = X_centered[:, :, premovement_mask]
premovement_times = times[premovement_mask]

# Make projection
P_neural = P_FR @ V_M.T
# Arrange as columns
X_premovement_matrix = X_premovement.reshape(182, -1)
premovement_coords = P_neural @ X_premovement_matrix

premovement_trajectories = premovement_coords.reshape(2, 108, 66)

fig = plot_premovement_comparison(
    FR_plot_trajectories,
    premovement_trajectories
)
fig.savefig("outputs/e6_premovement_comparison.pdf", bbox_inches="tight")

############
# Exercise 7
############

X_distorted = X.copy()
t0 = np.flatnonzero(times == -150).item() # index of time bin corresponding to -150 ms

rng = np.random.default_rng(42)

n_neurons, n_conditions, n_times = X.shape

# Select half of conditions for each neuron, distorting trajectories after t0 by reflecting across value at t0
for neuron in range(n_neurons):
    selected = rng.choice(
        n_conditions,
        size=n_conditions // 2,
        replace=False
    )

    boundary_values = X_distorted[neuron, selected, t0:t0 + 1]

    X_distorted[neuron, selected, t0:] = (
        2 * boundary_values
        - X_distorted[neuron, selected, t0:]
    )

# Normalize
control_maxima = X_distorted.max(axis=(1, 2), keepdims=True)
control_minima = X_distorted.min(axis=(1, 2), keepdims=True)
X_control_normalized = (X_distorted - control_minima) / (control_maxima - control_minima + 5)

# Mean centering
control_condition_mean = X_control_normalized.mean(axis=1, keepdims=True)
X_control_centered = X_control_normalized - control_condition_mean

# Select interval
X_control_movement = X_control_centered[:, :, movement_mask]
# Arrange as columns
X_control_matrix = X_control_movement.reshape(182, -1)

# Compute cov matrix
control_covariance = (X_control_matrix @ X_control_matrix.T) / X_control_matrix.shape[1]

# Find eigenvals and eigenvecs
control_eigenvals, control_eigenvecs = np.linalg.eigh(control_covariance)
control_order = np.argsort(control_eigenvals)[::-1]

# Get 12 dirs with largest variance
control_V_M = control_eigenvecs[:, control_order[:12]]

# Project
control_Z = control_V_M.T @ X_control_matrix
control_Z_trajectories = control_Z.reshape(12, 108, 46)

# Find fastest rotation plane
A_control = estimate_A(control_Z_trajectories)
control_rotation_values, control_rotation_vectors = np.linalg.eig(A_control)
control_fastest_index = np.argmax(control_rotation_values.imag)
control_fastest_vector = control_rotation_vectors[:, control_fastest_index]

# Make projection axes
control_real_axis = (control_fastest_vector.real / np.linalg.norm(control_fastest_vector.real))
control_imag_axis = (control_fastest_vector.imag / np.linalg.norm(control_fastest_vector.imag))
P_control_FR = np.vstack((control_real_axis, control_imag_axis))

# Project PC coords
control_FR_coords = P_control_FR @ control_Z
control_FR_trajectories = control_FR_coords.reshape(2, 108, 46)
control_FR_plot = control_FR_trajectories[:, :, FR_plot_mask]

# Plot and save
fig = plot_pc_trajectories(
    control_FR_plot,
    title="Control: Fastest-rotation plane (distorted data)",
    xlabel="Control FR axis 1",
    ylabel="Control FR axis 2"
)
fig.savefig("outputs/e7_control_rotation.pdf", bbox_inches="tight")

plt.show()