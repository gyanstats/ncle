import torch.multiprocessing as mp
import os
import torch
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import time
import pickle
import sys

COVERAGE_MODE = True

# Parameters
T = int(sys.argv[1])
mu_0, omega_0, alpha_0, beta_0 = 0.5, 0.1, 0.1, 0.8 # true parameter values
theta_0 = np.array([mu_0, omega_0, alpha_0, beta_0], dtype=np.float32)

batch_len = int(sys.argv[2])
num_batches = int(T / batch_len)

# Load the required batch
with open(f'train_l{batch_len}_1.pkl', 'rb') as f:
    nle_batches = pickle.load(f)

# Extract likelihood estimator for the specific batch size
likelihood_estimator = nle_batches['likelihood estimator']
# Convert to float64
likelihood_estimator = likelihood_estimator.double()

torch.set_num_threads(1)  # Keep this to avoid oversubscription
max_workers = 4 # adjust based on SLURM cpus-per-task
max_epochs = 8000
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
    epsilon[0] = np.random.normal(theta[0], np.sqrt(sigma_sq[0]))  # simulate initial epsilon

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

def training_loop_ncl(max_epochs, x_0, num_batches):

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
        context = raw.expand(batches_tensor.shape[0], 4)
        log_p = likelihood_estimator.log_prob(batches_tensor, context).sum()
        loss = -log_p
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
    
    
def calculate_scores_ncl(mcle, num_batches):
    # Convert to original params for simulation
    theta_orig = reparam_to_original(mcle).detach().numpy()
    x = simulator(theta=theta_orig, T=T)

    x_tensor = torch.from_numpy(x)
    batches = torch.chunk(x_tensor, num_batches)
    batches_tensor = torch.stack(batches)

    def log_p_b(x_b, mcle_):
        return likelihood_estimator.log_prob(x_b.unsqueeze(0), context=mcle_.unsqueeze(0)).squeeze()

    grad_log_p_b = torch.func.grad(log_p_b, argnums=1)
    scores = torch.vmap(grad_log_p_b, in_dims=(0, None))(batches_tensor, mcle)
    return scores
    
# Define function for parallelisation
def task_score(args):
    mcle_detached, num_batches = args
    mcle_detached = mcle_detached.clone().detach().requires_grad_(True)
    score_tensor = calculate_scores_ncl(mcle_detached, num_batches)
    return score_tensor.detach()

# Estimate sensitivity and variability matrices
def calculate_S_and_V_ncl(mcle, num_batches, n):
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
    
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    if COVERAGE_MODE:
        x_0 = simulator(theta=theta_0, T=T)
    else:
        np.random.seed(0)
        x_0 = simulator(theta=theta_0, T=T)
        print(x_0) # to check that the seed worked
    x_0_tensor = torch.from_numpy(x_0)

    ci_start = time.time()

    print(f"Starting training loop...", flush=True)
    mcle_start = time.time()
    mcle = training_loop_ncl(max_epochs, x_0_tensor, num_batches)
    mcle_end = time.time()
    mcle_time = mcle_end - mcle_start
    mcle_orig = reparam_to_original(mcle.detach())
    print(f"Finished training loop in {mcle_time:.2f} seconds. MCLE (reparam): {mcle}.", flush=True)
    print(f"MCLE (original): {mcle_orig}", flush=True)

    print(f"Started calculation of GIM...", flush=True)
    gim_start = time.time()
    S, V = calculate_S_and_V_ncl(mcle, num_batches, n)
    G = S @ torch.linalg.inv(V) @ S
    gim_end = time.time()
    gim_time = gim_end - gim_start
    print(f"Finished calculation of GIM in {gim_time:.2f} seconds.", flush=True)

    variances = torch.diagonal(torch.linalg.inv(G))
    se = torch.sqrt(variances)
    z = norm.ppf(1 - (1 - signif_level) / 2)

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
        filename = f'ci_T{N_string(T)}_l{batch_len}_task{task_id}.pkl'
    else:
        filename = f'ci_T{N_string(T)}_l{batch_len}.pkl'
        
    with open(filename, 'wb') as f:
        pickle.dump(ci, f)

