import torch.multiprocessing as mp
import os
import torch
import numpy as np
from scipy.stats import norm
import time
import pickle
import sys

COVERAGE_MODE = False

# Parameters
T = int(sys.argv[1]) # Sequence length
phi_0 = 0.8 # True value

batch_len = int(sys.argv[2]) # Batch length
num_batches = int( T/batch_len ) # Number of batches

# For pickling
def N_string(N):
	if N >= 1_000_000: return f'{N//1_000_000}m'
	if N >= 1_000:     return f'{N//1_000}k'
	return str(N)

torch.set_num_threads(1)  # Keep this to avoid oversubscription
max_workers = 4 # adjust based on SLURM cpus-per-task
max_epochs = 4000
n = 100 # number of summands in S and V calculations
signif_level = 0.95

# Simulator
def simulator(phi, T):
    # Convert phi to float if a Tensor is passed in
    if isinstance(phi, torch.Tensor):
        phi = phi.detach().item()

    X = np.zeros(T, dtype=np.float32)
    X[0] = np.random.normal(0, scale=np.sqrt(1 / (1 - phi**2)))
    for t in range(1, T):
        X[t] = phi * X[t - 1] + np.random.normal(0, 1)

    return torch.from_numpy(X)  # Returns torch.float32 tensor

def ar1_log_lik(x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    T = x.shape[0]
    phi = phi.squeeze()
    var0 = 1 / (1 - phi**2)
    
    log_p = -0.5 * torch.log(2 * torch.pi * var0) - 0.5 * x[0]**2 / var0
    residuals = x[1:] - phi * x[:-1]
    log_p += -0.5 * (T - 1) * torch.log(torch.tensor(2 * torch.pi))
    log_p += -0.5 * torch.sum(residuals**2)
    
    return log_p

# Vectorised, for training loop
def ar1_log_lik_vectorised(x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Vectorised log-likelihood for AR(1) model over batches.

    Args:
        x: (B, T)
        phi: (1,)

    Returns:
        log-likelihood: (B,)
    """
    B, T = x.shape
    phi = phi.squeeze()

    var0 = 1 / (1 - phi**2)
    log_p0 = -0.5 * torch.log(2 * torch.pi * var0) - 0.5 * x[:, 0]**2 / var0

    residuals = x[:, 1:] - phi * x[:, :-1]
    log_p_rest = -0.5 * (T - 1) * torch.log(torch.tensor(2 * torch.pi)) - 0.5 * torch.sum(residuals**2, dim=1)

    return log_p0 + log_p_rest

# Optimisation procedure for finding MCLE
def training_loop_cl(max_epochs, x_0, num_batches):

    phi = torch.tensor([0.1], requires_grad=True) # initialise phi
    optimizer = torch.optim.Adam([phi])
    
    # Split x_0 into batches
    #batches = np.array_split(x_0, num_batches)
    #batches_mat = np.vstack(batches)
    batches = tuple(chunk for chunk in torch.chunk(x_0, num_batches))

    # Track for early stopping
    prev_loss = float('inf')
    
    # Optimisation
    for epoch in range(max_epochs):
        # Calculate loss with this phi
        batches_tensor = torch.stack(batches) # shape: (B, T)
        log_p = ar1_log_lik_vectorised(batches_tensor, phi).sum()

        loss = -log_p # loss is the negative log-likelihood given x_0
        
        optimizer.zero_grad() # reset gradients to zero
        loss.backward()

        optimizer.step() # step to the next phi
        
        # Apply constraints to phi
        with torch.no_grad():
            phi.clamp_(-0.999, 0.999)
        
        # Check for convergence
        current_loss = loss.item()
        
        # Early stopping if no improvement
        if abs(prev_loss - current_loss) < 1e-9:
            print(f"Converged at epoch {epoch}: loss stopped changing")
            break
        prev_loss = current_loss
        
        # Print progress every 1000 epochs
        if epoch % 200 == 0:
            print(f"Epoch {epoch}/{max_epochs}, Loss: {loss.item()}, phi: {phi.data}")

    return phi

# Function for calculating score for each batch, i.e. grad log(p_b) for each b
def calculate_scores_cl(mcle: torch.Tensor, num_batches: int) -> torch.Tensor:
    """
    Compute gradient (score) for each batch.

    Args:
        mcle: (1,) MCLE estimate
        num_batches: number of batches

    Returns:
        Tensor of shape (num_batches, 1)
    """
    # Simulate data
    x = simulator(phi=mcle.detach(), T=T)

    # Split x into batches and stack them
    batches = torch.chunk(x, num_batches)
    batches_tensor = torch.stack(batches) # shape: (num_batches, batch_len)

    # Function for calculating log sub-likelihood
    def log_p_b(x_b, mcle_):
        return ar1_log_lik(x_b, mcle_).squeeze() # convert shape [1] to shape ()

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

# Calculate Godambe
def calculate_G_cl(mcle, num_batches, n):
    S, V = calculate_S_and_V_cl(mcle, num_batches, n) # calculate S and V approximation
    G = S @ torch.inverse(V) @ S # calculate Godambe
    return G

# Calculate CI
def calculate_ci_cl(signif_level, num_batches, n):

    ci_start = time.time()
    # Calculate mcle given x_0
    print(f"Starting training loop...", flush=True)
    mcle_start = time.time()
    mcle = training_loop_cl(max_epochs, x_0, num_batches)
    mcle_end = time.time()
    mcle_time = mcle_end - mcle_start
    print(f"Finished training loop in {mcle_time:.2f} seconds. MCLE: {mcle}.", flush=True)

    ''' Calculate CI '''
    print(f"Starting calculation of GIM...", flush=True)
    gim_start = time.time()
    G = calculate_G_cl(mcle, num_batches, n) # calculate Godambe
    gim_end = time.time()
    gim_time = gim_end - gim_start
    print(f"Finished calculation of GIM in {gim_time:.2f} seconds.", flush=True)
    
    variances = torch.diagonal(torch.inverse(G)) # extract diagonal elements of covariance matrix to get the variances
    se = torch.sqrt(variances) # (asymptotic) standard error of the MCLE
    z = norm.ppf(1 - (1 - signif_level) / 2) # z-score
    
    lower_bound = mcle - z * se
    upper_bound = mcle + z * se
    ci_end = time.time()
    ci_time = ci_end - ci_start

    return {
        "x_0": x_0,
        "mcle": mcle.item(),
        "gim": G.detach().numpy(),
        "ci_bounds": (lower_bound.item(), upper_bound.item()),
        "mcle_time": mcle_time,
        "gim_time": gim_time,
        "ci_time": ci_time
    }
    
# Wrap the main execution code with this guard
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

	# Generate observed dataset x_0
    if COVERAGE_MODE:
        x_0 = simulator(phi=phi_0, T=T)
    else:
        np.random.seed(0)
        x_0 = simulator(phi=phi_0, T=T)

    # Calculate a CI
    ci = calculate_ci_cl(signif_level, num_batches, n)

    # Save results with unique filenames
    if COVERAGE_MODE:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
        filename = f'results_ncle/analyt/ci_T{N_string(T)}_l{batch_len}_task{task_id}.pkl'
    else:
        filename = f'results_ncle/analyt/ci_T{N_string(T)}_l{batch_len}.pkl'

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(ci, f)
