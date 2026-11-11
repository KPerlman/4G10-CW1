import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_individual_psths, plot_population_mean_psth, plot_neuron_maxima, plot_pc_trajectories

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

###
# a
###

# Derivation of full-log likelyhood
# r_ct = (z_ct+1 - z_ct) - Az_ct
# Note z_ct+1 - z_ct is delta between time bins, A is transition matrix, z_ct is state at time t for condition c
# 46 time bins after masking, 45 transitions per condition, 4860 transitions total
# Gaussian distribution formula where d is observed change and m is predicted change (=A_z) and sigma = 1,
# p(d|m) = (1/sqrt(2*pi)) * exp(-((d-m)^2)/2)
# Taking log gives log(p(d|m)) = -0.5 * log(2*pi) - 0.5 * (d-m)^2
# Note first term is constant and second is one squared-residual term
# Multiply noise cordinate densities (as they are independent), given by p(d|m) = product of p(d|m) over all transitions
# Taking log gives log(p(d|m)) = -M/2 * log(2*pi) - 0.5 * sum((d-m)^2) where M is number of PCs
# ||r||^2 defined to be sum((d-m)^2) seen in previous line
# Each condition has T - 1 transitions, C conditions, so M = C * (T - 1)
# Final function is log(p(d|m)) = -C*(T-1)/2 * log(2*pi) - 0.5 * ||r_ct||^2 where r_ct = (z_ct+1 - z_ct) - Az_ct
# Final function also referred to as log-likelihood function, l(A)
# Note three sums are actually present, one over conditions, one over time bins (T-2 as its inclusive), and one over state dimensions (12)
# Overall, log-likelihood function measures how well proposed model explains observed data and -l(A) is cost function to be minimized to find ideal A
# Final eqns: R = D - AZ_, l(A) = constant - 0.5 * ||R||^2, and gradient of l(A) = RZ_^T

Z_current = Z_trajectories[:, :, :-1] # Z at time t for all conditions so its states are where valid transitions occur
delta_Z = np.diff(Z_trajectories, axis=2)

Z_current_matrix = Z_current.reshape(12, -1)
delta_Z_matrix = delta_Z.reshape(12, -1)

# log-likelihood at A_trial
A_trial = np.zeros((12, 12))
residuals = delta_Z_matrix - A_trial @ Z_current_matrix # actual change - predicted change
squared_error = np.sum(residuals ** 2)

n_residuals = residuals.size
log_likelihood = (
    -0.5 * n_residuals * np.log(2 * np.pi)
    -0.5 * squared_error
)

# Derivation of gradient of log-likelihood function
# How changes in A affect log-likelihood function
# With a predicted change m = sum(Az_ct), the residual r_ct=(z_ct+1-z_ct)-m contributes -0.5*(r_ct^2) to log-likelihood
# del_l/del_A(-0.5*(r_ct^2)) = r_ct*z_ct
# For whole dataset, del_l/del_A = sum(r_ct*z_ct) over all conditions and time bins
# So gradient is = (D - AZ)Z^T excluding last time bin where D = Z[:, 1:] - Z[:, :-1] and Z = Z[:, :-1]

gradient = residuals @ Z_current_matrix.T
# how does log-likelihood change if increase entry in row 0, column 1 slightly in A

epsilon = 1e-6

# make copy of A_trial and slightly change (0, 1) entry by epsilon in + and - dirs
A_plus = A_trial.copy()
A_minus = A_trial.copy()
A_plus[0, 1] += epsilon
A_minus[0, 1] -= epsilon

# get new prediction errors
residuals_plus = delta_Z_matrix - A_plus @ Z_current_matrix
residuals_minus = delta_Z_matrix - A_minus @ Z_current_matrix

score_plus = -0.5 * np.sum(residuals_plus ** 2)
score_minus = -0.5 * np.sum(residuals_minus ** 2) # just the negative of squared error term in log-likelihood function as constant term is irrelevant for gradient

# central finite difference approx of gradient
numerical_gradient = (score_plus - score_minus) / (2 * epsilon)

print("Numerical: ", numerical_gradient)
print("Analytical:", gradient[0, 1])

###
# b
###

# A must be antisymmetric, so number of independent parameters is 12*11/2 = 66, so construction of A is just upper triangular part of A and then copy to lower triangular part with a negative sign
# Will call vector of parameters beta, vector of length 66 containing independent parameters of A
# Using template matricies to create A from beta, ie A = aH1 + bH2 + cH3 + ... where H1, H2, H3 are template matrices with 1 in appropriate upper triangular position and -1 in corresponding lower triangular position and 0 elsewhere

