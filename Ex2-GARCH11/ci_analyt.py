import torch.multiprocessing as mp
import os
import torch
import numpy as np
from torch.distributions.normal import Normal  # (or use torch.distributions.Normal directly)
from scipy.stats import norm
from scipy.optimize import minimize
import time
import pickle
import sys

COVERAGE_MODE = True

# Parameters
T = int(sys.argv[1]) # Adjust this when necessary
mu_0, omega_0, alpha_0, beta_0 = 0.5, 0.1, 0.1, 0.8 # true parameter values
theta_0 = torch.tensor([mu_0, omega_0, alpha_0, beta_0], dtype=torch.float32)

batch_len = int(sys.argv[2]) # adjusts automatically
num_batches = T//batch_len

torch.set_num_threads(1)  # Keep this to avoid oversubscription
max_workers = 4 # adjust based on SLURM cpus-per-task
n = 100 # number of summands in S and V calculations
signif_level = 0.95

# For pickling at the end of the script
def N_string(N):
    if N >= 1_000_000: return f'{N//1_000_000}m'
    if N >= 1_000:     return f'{N//1_000}k'
    return str(N)

# Simulator
def simulator(theta, T, burn_in=500):
    T_total = T + burn_in
    sigma_sq = np.zeros(T_total, dtype=np.float32)  # initialise
    sigma_sq[0] = theta[1] / (1 - theta[2] - theta[3])  # unconditional variance init
    epsilon = np.zeros(T_total, dtype=np.float32)  # initialise
    epsilon[0] = np.random.normal(theta[0], np.sqrt(sigma_sq[0]))  # simulate initial$

    # Simulate GARCH(1,1) process
    for t in range(1, T_total):
        sigma_sq[t] = theta[1] + theta[2]*(epsilon[t-1]-theta[0])**2 + theta[3]*sigma_sq[t-1]
        epsilon[t] = np.random.normal(theta[0], np.sqrt(sigma_sq[t]))

    return epsilon[burn_in:]
    
def reparam_to_original(theta_r):
    """Convert (mu, sigma2, persistence, alpha_frac) -> (mu, omega, alpha, beta), torch-compatible."""
    mu          = theta_r[0]
    sigma2      = theta_r[1]
    persistence = theta_r[2]
    alpha_frac  = theta_r[3]
    omega = sigma2 * (1 - persistence)
    alpha = alpha_frac * persistence
    beta  = (1 - alpha_frac) * persistence
    return torch.stack([mu, omega, alpha, beta])

def quasi_ll(x, theta):
    T = len(x)
    mu, alpha0, alpha1, beta1 = theta

    # Initialize the first variance
    sigma_sq_0 = alpha0 / (1 - alpha1 - beta1)
    sigma_sq_list = [sigma_sq_0]

    # Recursively compute all sigma_sq values without inplace operations
    for t in range(1, T):
        prev_sigma_sq = sigma_sq_list[-1]
        new_sigma_sq = alpha0 + alpha1 * (x[t-1] - mu)**2 + beta1 * prev_sigma_sq
        sigma_sq_list.append(new_sigma_sq)

    sigma_sq = torch.stack(sigma_sq_list)

    # Compute log likelihood manually (elementwise normal log-prob)
    log_p = -0.5 * torch.log(2 * torch.pi * sigma_sq) - (x - mu)**2 / (2 * sigma_sq)

    return log_p.sum()


def quasi_ll_batch(batches_tensor: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Vectorised quasi log-likelihood for GARCH(1,1) model over batches.

    Args:
        batches_tensor: (B, T) data
        theta: (4,) parameter [mu, omega, alpha, beta]

    Returns:
        Tensor of shape (B,) with log-likelihood per batch
    """
    B, T = batches_tensor.shape
    mu, omega, alpha, beta = theta

    x = batches_tensor
    sigma_sq = torch.zeros((B, T), dtype=x.dtype)

    # Initial variance
    sigma_sq[:, 0] = omega / (1 - alpha - beta)
    log_lik = -0.5 * torch.log(2 * torch.pi * sigma_sq[:, 0]) - 0.5 * (x[:, 0] - mu)**2 / sigma_sq[:, 0]

    # Recompute sigma_sq in a loop with clone() to preserve computation graph
    for t in range(1, T):
        sigma_sq_new = sigma_sq.clone()
        sigma_sq_new[:, t] = omega + alpha * (x[:, t-1] - mu)**2 + beta * sigma_sq[:, t-1]
        sigma_sq = sigma_sq_new
        log_lik += -0.5 * torch.log(2 * torch.pi * sigma_sq[:, t]) - 0.5 * (x[:, t] - mu)**2 / sigma_sq[:, t]

    return log_lik



# Fusion, using torch
def cll(batches: tuple, theta: torch.Tensor, num_batches: int) -> torch.Tensor:
    """
    Compute total log-likelihood summed over all batches.

    Args:
        batches: tuple of (batch_len,) tensors (each a batch of the time series)
        theta: (4,) parameter
        num_batches: number of batches

    Returns:
        Total log-likelihood (scalar tensor)
    """
    return sum(quasi_ll(batch, theta) for batch in batches)

# Optimisation procedure for finding MCLE, using L-BFGS-B (scipy).
def training_loop_cl(x_0, num_batches):

    # Split x_0 into batches. Optimise in float64 for improved precision.
    batches = torch.chunk(x_0, num_batches)
    batches_tensor = torch.stack(batches).to(torch.float64)  # shape: (num_batches, batch_len)

    # Initial parameters (mu, sigma2, persistence, alpha_frac)
    x0 = np.array([0.5, 1.0, 0.85, 0.1 / 0.85], dtype=np.float64)

    # Per-coordinate box bounds
    bounds = [
        (-1 + 1e-4, 1 - 1e-4),   # mu in (-1, 1)
        (1e-6, None),            # sigma2 > 0
        (1e-4, 1 - 1e-4),        # persistence in (0, 1)  =>  alpha + beta < 1
        (1e-4, 1 - 1e-4),        # alpha_frac in (0, 1)
    ]

    # Objective returning (loss, gradient) for scipy (jac=True).
    def f_and_g(raw_np):
        raw = torch.tensor(raw_np, dtype=torch.float64, requires_grad=True)
        theta_orig = reparam_to_original(raw)
        loss = -quasi_ll_batch(batches_tensor, theta_orig).sum()
        loss.backward()
        return loss.item(), raw.grad.detach().numpy()

    # Convergence is governed by the tolerances; maxiter is only a safety cap
    # (L-BFGS-B converges in tens of iterations on this smooth 4-parameter problem).
    result = minimize(
        f_and_g, x0, jac=True, method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
    )

    print(f"L-BFGS-B finished: success={result.success}, message={result.message}, "
          f"nit={result.nit}, nfev={result.nfev}, fun={result.fun}", flush=True)

    # Convert MCLE back to float32
    mcle = torch.tensor(result.x, dtype=torch.float32, requires_grad=True)
    return mcle
    
def calculate_scores_cl(mcle, num_batches):
    # Convert to original params for simulation
    mcle_orig = reparam_to_original(mcle).detach().numpy()
    x = simulator(theta=mcle_orig, T=T) # simulate x

    # Split x into batches and stack them
    x_tensor = torch.from_numpy(x)
    batches = torch.chunk(x_tensor, num_batches)
    batches_tensor = torch.stack(batches)

    # Function for calculating log sub-likelihood
    def log_p_b(x_b, mcle_):
        return quasi_ll(x_b, reparam_to_original(mcle_)).squeeze() # convert shape [1] to shape ()

    # Get score of a single batch (i.e. gradient of log sub-likelihood) w.r.t. mcle
    def grad_log_p_b(x_b, mcle_):
        return torch.func.grad(log_p_b, argnums=1)(x_b, mcle_)

    # Vectorise over batches (i.e. over each row of 'batches_tensor')
    scores = torch.vmap(grad_log_p_b, in_dims=(0, None))(batches_tensor, mcle)
    return scores # tensor of gradients for each batch
    
# Define function for parallelisation
def task_score(args):
    mcle_detached, num_batches = args
    mcle_detached = mcle_detached.clone().detach().requires_grad_(True)
    score_tensor = calculate_scores_cl(mcle_detached, num_batches)
    return score_tensor.detach()

# Estimate sensitivity and variability matrices
def calculate_S_and_V_cl(mcle, num_batches, n):
    args = [(mcle.clone().detach(), num_batches)] * n

    # Calculate scores for each of the n Monte Carlo samples in parallel
    with mp.Pool(processes=max_workers) as pool:
        scores_list = pool.map(task_score, args)

    scores = torch.stack(scores_list, dim=0) # shape: (n, num_batches, d)
    
    # Approximate S
    S = torch.einsum("nbi, nbj -> ij", scores, scores) / n
    # Approximate V
    sum_over_b = scores.sum(dim=1) # shape (n, d)
    V = torch.einsum("ni, nj -> ij", sum_over_b, sum_over_b) / n
    
    return (S, V)

    
# Calculate CI
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    ''' Simulate observed data then use that to get mcle '''
    if COVERAGE_MODE:
        x_0 = simulator(theta=theta_0, T=T)
    else:
        np.random.seed(0)
        x_0 = simulator(theta=theta_0, T=T)
        print(x_0) # to check that the seed worked
    x_0_tensor = torch.from_numpy(x_0)
    
    ci_start = time.time()
    # Calculate mcle given x_0
    print(f"Starting training loop...", flush=True)
    mcle_start = time.time()
    mcle = training_loop_cl(x_0_tensor, num_batches)  # in reparam space
    mcle_end = time.time()
    mcle_time = mcle_end - mcle_start
    mcle_orig = reparam_to_original(mcle.detach())
    print(f"Finished training loop in {mcle_time:.2f} seconds. MCLE (reparam): {mcle}.", flush=True)
    print(f"MCLE (original): {mcle_orig}", flush=True)

    ''' Calculate CI '''
    print(f"Starting calculation of GIM...", flush=True)
    gim_start = time.time()
    S, V = calculate_S_and_V_cl(mcle, num_batches, n) # calculate S and V approximation
    G = S @ torch.linalg.inv(V) @ S # calculate Godambe
    gim_end = time.time()
    gim_time = gim_end - gim_start
    print(f"Finished calculation of GIM in {gim_time:.2f} seconds.", flush=True)
    
    variances = torch.diagonal(torch.linalg.inv(G)) # extract diagonal elements of covariance matrix to get the variances
    se = torch.sqrt(variances) # (asymptotic) standard error of the MCLE
    z = norm.ppf(1 - (1 - signif_level) / 2) # z-score
    
    lower_bound = mcle - z * se
    upper_bound = mcle + z * se

    # Delta method
    mu, sigma2, p, af = mcle.detach()
    J = torch.tensor([
        # dmu   dsigma2      dp            daf
        [1,     0,           0,            0         ],  # mu
        [0,     (1-p),      -sigma2,       0         ],  # omega
        [0,     0,           af,           p         ],  # alpha
        [0,     0,           (1-af),      -p         ],  # beta
    ])
    cov_reparam = torch.linalg.inv(G)
    cov_original = J @ cov_reparam @ J.T
    se_original = torch.sqrt(torch.diagonal(cov_original))
    lower_orig = mcle_orig - z * se_original
    upper_orig = mcle_orig + z * se_original

    ci_end = time.time()
    ci_time = ci_end - ci_start

    ci = {
        "x_0": x_0,
        "mcle_reparam": mcle.detach(),
        "mcle_original": mcle_orig.detach(),
        "S": S.detach().numpy(),
        "V": V.detach().numpy(),
        "gim": G.detach().numpy(),
        "ci_bounds_reparam": {
            "mu":          (lower_bound[0].item(), upper_bound[0].item()),
            "sigma2":      (lower_bound[1].item(), upper_bound[1].item()),
            "persistence": (lower_bound[2].item(), upper_bound[2].item()),
            "alpha_frac":  (lower_bound[3].item(), upper_bound[3].item()),
        },
        "ci_bounds_original": {
            "mu":    (lower_orig[0].item(), upper_orig[0].item()),
            "omega": (lower_orig[1].item(), upper_orig[1].item()),
            "alpha": (lower_orig[2].item(), upper_orig[2].item()),
            "beta":  (lower_orig[3].item(), upper_orig[3].item()),
        },
        "mcle_time": mcle_time,
        "gim_time": gim_time,
        "ci_time": ci_time
    }

    # Save results with unique filenames
    if COVERAGE_MODE:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
        filename = f'analyt_ci_T{N_string(T)}_l{batch_len}_task{task_id}.pkl'
    else:
        filename = f'analyt_ci_T{N_string(T)}_l{batch_len}.pkl'
        
    with open(filename, 'wb') as f:
        pickle.dump(ci, f)
