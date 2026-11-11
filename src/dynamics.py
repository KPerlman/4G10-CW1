import numpy as np

def construct_H(M):
    K = M * (M - 1) // 2 # number of independent params in antisymmetric matrix
    H = np.zeros((K, M, M)) # K template matrices of size M x M

    a = 0
    for i in range(M):
        for j in range(i + 1, M): # just upper triangular part of A
            H[a, i, j] = 1
            H[a, j, i] = -1
            a += 1

    return H

def estimate_A(Z):
    M, C, T = Z.shape

    # E4(a) Prepare current states and observed changes

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
    # Each condition has T - 1 transitions, C conditions, so total transitions is C * (T - 1)
    # Final function is log(p(d|m)) = -MC*(T-1)/2 * log(2*pi) - 0.5 * sum_ct(||r_ct||^2) where r_ct = (z_ct+1 - z_ct) - Az_ct
    # Final function also referred to as log-likelihood function, l(A)
    # Note three sums are actually present, one over conditions, one over time bins (T-2 as its inclusive), and one over state dimensions (12)
    # Overall, log-likelihood function measures how well proposed model explains observed data and -l(A) is cost function to be minimized to find ideal A
    # Final eqns: R = D - AZ_, l(A) = constant - 0.5 * ||R||^2, and gradient of l(A) = RZ_^T

    # Derivation of gradient of log-likelihood function
    # How changes in A affect log-likelihood function
    # With a predicted change m = sum(Az_ct), the residual r_ct=(z_ct+1-z_ct)-m contributes -0.5*(r_ct^2) to log-likelihood
    # del_l/del_A(-0.5*(r_ct^2)) = r_ct*z_ct
    # For whole dataset, del_l/del_A = sum(r_ct*z_ct) over all conditions and time bins
    # So gradient is = (D - AZ)Z^T excluding last time bin where D = Z[:, 1:] - Z[:, :-1] and Z = Z[:, :-1]

    Z_current = Z[:, :, :-1] # Z at time t for all conditions so its states are where valid transitions occur
    delta_Z = np.diff(Z, axis=2)
    Z_current_matrix = Z_current.reshape(M, -1)
    delta_Z_matrix = delta_Z.reshape(M, -1)

    # E4(b) Construct antisymmetric placement templates

    # A must be antisymmetric, so number of independent parameters is 12*11/2 = 66, so construction of A is just upper triangular part of A and then copy to lower triangular part with a negative sign
    # Will call vector of parameters beta, vector of length 66 containing independent parameters of A
    # Using template matricies to create A from beta, ie A = aH1 + bH2 + cH3 + ... where H1, H2, H3 are template matrices with 1 in appropriate upper triangular position and -1 in corresponding lower triangular position and 0 elsewhere

    H = construct_H(M)
    K = H.shape[0] # number of independent parameters in antisymmetric matrix

    # E4(c) Calculate W, b, Q

    # Already have unconstrained gradient G = ▽l(A) = (D - AZ)Z^T, so can use chain rule to get gradient wrt beta
    # Each Gij says how much log-likelihood changes if change Aij slightly,so just need to sum over all Gij that depend on each independent parameter in beta
    # del(l)/del(beta_0) = G01 - G10 as each beta controls a pair of entries in A
    # del(l)/del(beta_a) = Gij - Gji
    # W_a = H_aZ_current from coursework
    # Each W_a holds one param's worth of contribution to predicted change AZ, so can use W_a to get gradient wrt beta
    # (AZ_current)_ij = sum_a(beta_a * W_ain)
    # For one transition n, predicted change in coord i is m_in = sum_b(beta_b * W_bin) where b runs through all params
    # If adjust only one param beta_a, effect is del(m_in)/del(beta_a) = W_ain
    # Because residual r_in = d_in - m_in, del(r_in)/del(beta_a) = -W_ain
    # So final result, differentiating log-likelihood contribution wrt beta_a gives del(l)/del(beta_a) = sum_in(r_in * W_ain)
    # But have to get in form of paper
    # I just did this by hand so we'll just outline that in the report
    # Form given is ▽beta(l) = b - betaQ
    # b_a = sum_in(d_in * W_ain) and Q_qa = sum_in(W_qin * W_ain)

    W = np.tensordot(H, Z_current_matrix, axes=([2], [0])) # Gives H_a * Z_current for each a
    # Reshape W to get b and Q
    W_matrix = W.reshape(K, -1)
    delta_flat = delta_Z_matrix.reshape(-1)
    b = delta_flat @ W_matrix.T
    Q = W_matrix @ W_matrix.T

    # E4(d) Solve for fitted parameters, construct A
    beta_hat = np.linalg.solve(Q.T, b)
    A_hat = np.tensordot(beta_hat, H, axes=1)

    return A_hat
