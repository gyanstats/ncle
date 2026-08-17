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

num_sims = int(sys.argv[2]) # Number of training simulations
batch_len = int(sys.argv[3]) # Batch length
num_batches = int( T/batch_len ) # Number of batches

# For pickling
def N_string(N):
	if N >= 1_000_000: return f'{N//1_000_000}m'
	if N >= 1_000:     return f'{N//1_000}k'
	return str(N)
    
# Load the required batch
with open(f'results_ncle/nle_summaries_N{N_string(num_sims)}/train_l{batch_len}.pkl', 'rb') as f:
    nle_batches = pickle.load(f)

# Extract likelihood estimator for the specific batch size
likelihood_estimator = nle_batches['likelihood estimator']

torch.set_num_threads(1)  # Keep this to avoid oversubscription
max_workers = 4 # adjust based on SLURM cpus-per-task
max_epochs = 4000
n = 100 # number of summands in S and V calculations
signif_level = 0.95

# Simulator
def simulator(phi: torch.Tensor, T: int) -> torch.Tensor:
    """
    Simulate AR(1) process using torch only.
    Args:
        phi (torch.Tensor): shape (1,)
        T (int): time series length

    Returns:
        torch.Tensor: shape (T,)
    """
    x = torch.empty(T)
    eps = torch.randn(T)
    x[0] = eps[0] * torch.sqrt(torch.tensor(1.0) / (1 - phi**2))

    for t in range(1, T):
        x[t] = phi*x[t-1] + eps[t]

    return x

def calculate_sufficient_stats(x: torch.Tensor) -> torch.Tensor:
    """
    Compute sufficient statistics for AR(1) model for multiple time series.
    
    Args:
        x (torch.Tensor): shape (N, T) for N series, or (T,) for single series
    
    Returns:
        torch.Tensor: shape (N, 2) containing [s1, s2] for each series
            s1 = sum_{t=2}^{T-1} x_t^2  
            s2  = sum_{t=2}^{T} x_{t-1} * x_t
    """
    # Handle both single series and multiple series
    if x.dim() == 1:
        # Single series: shape (T,)
        T = x.shape[0]
        s1 = torch.sum(x[1:-1]**2)
        s2 = torch.sum(x[:-1] * x[1:])
        stats = torch.stack([s1, s2])
    else:
        # Multiple series: shape (N, T)
        N, T = x.shape
        s1 = torch.sum(x[:, 1:-1]**2, dim=1)  # shape: (N,)
        s2 = torch.sum(x[:, :-1] * x[:, 1:], dim=1)  # shape: (N,)
        stats = torch.stack([s1, s2], dim=1)
    
    return stats

# Optimisation procedure for finding MCLE
def training_loop_ncl(max_epochs, x_0, num_batches):
    '''
    Makes the parameters in .log_prob() compatible with the function.
    Then calculate score for each batch.
    '''
    phi = torch.tensor([0.1], requires_grad=True)
    optimizer = torch.optim.Adam([phi])
    
    # Split x_0 into batches and compute sufficient statistics
    batches = torch.chunk(x_0, num_batches)
    batches_tensor = torch.stack(batches)  # shape: (num_batches, batch_len)
    s_batches = calculate_sufficient_stats(batches_tensor) # shape: (num_batches, 3)
    
    # Track for early stopping
    prev_loss = float('inf')

    # Optimisation
    for epoch in range(max_epochs):
        # Reshape phi for batch processing
        context = phi.expand(s_batches.shape[0], 1).float()
        
        # Calculate composite log-likelihood using sufficient statistics
        log_p = likelihood_estimator.log_prob(s_batches, context).sum()
        loss = -log_p
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Apply hard constraints to phi after optimisation step
        with torch.no_grad():
            phi.clamp_(-0.999, 0.999)
        
        # Check for convergence
        current_loss = loss.item()
        
        if abs(prev_loss - current_loss) < 1e-9:
            print(f"Converged at epoch {epoch}: loss stopped changing")
            break
        prev_loss = current_loss
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}/{max_epochs}, Loss: {loss.item()}, phi: {phi.data}")
        
    return phi
    
def calculate_scores_ncl(mcle, num_batches):
    '''
    Makes the parameters in .log_prob() compatible with the function.
    Then calculate score for each batch.
    '''
    # Simulate data
    x = simulator(phi=mcle.detach().item(), T=T)
    
    # Split x into batches and compute sufficient statistics
    batches = torch.chunk(x, num_batches)
    batches_tensor = torch.stack(batches)  # shape: (num_batches, batch_len)
    s_batches = calculate_sufficient_stats(batches_tensor)  # shape: (num_batches, 3)

    # Function for calculating log sub-likelihood
    def log_p_b(s_b, mcle_):
        '''
        Before calculating .log_prob(),
        converts s_b from shape (3,) to (1, 3) and converts mcle from shape (1,) to (1, 1).
        Then applies .squeeze() to the result to convert it to shape ().
        '''
        return likelihood_estimator.log_prob(s_b.unsqueeze(0), context=mcle_.unsqueeze(0)).squeeze()

    # Get score of a single batch (i.e. gradient of log sub-likelihood) w.r.t. mcle
    grad_log_p_b = torch.func.grad(log_p_b, argnums=1)

    # Vectorise over batches (i.e. over each row of 's_batches')
    scores = torch.vmap(grad_log_p_b, in_dims=(0, None))(s_batches, mcle)

    return scores  # tensor of gradients for each batch

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
    
# Calculate Godambe
def calculate_G_ncl(mcle, num_batches, n):
    S, V = calculate_S_and_V_ncl(mcle, num_batches, n) # calculate S and V approximation
    G = S @ torch.inverse(V) @ S # calculate Godambe
    return G
    
# Calculate CI
def calculate_ci_ncl(signif_level, num_batches, n):

    ''' Simulate observed data then use that to get mcle '''
    x_0 = simulator(phi=phi_0, T=T) # simulate x_0 (not seeded)
    
    ci_start = time.time()
    # Calculate mcle given x_0
    print(f"Starting training loop on GPU...", flush=True)
    mcle_start = time.time()
    mcle = training_loop_ncl(max_epochs, x_0, num_batches)
    mcle_end = time.time()
    mcle_time = mcle_end - mcle_start
    print(f"Finished training loop in {mcle_time:.2f} seconds. MCLE: {mcle}.", flush=True)

    ''' Calculate CI '''
    print(f"Starting calculation of GIM on CPU...", flush=True)
    gim_start = time.time()
    G = calculate_G_ncl(mcle, num_batches, n) # calculate Godambe
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

    # Calculate a CI
    ci = calculate_ci_ncl(signif_level, num_batches, n)

    # Save results with unique filenames
    if COVERAGE_MODE:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
        filename = f'results_ncle/nle_summaries_N{N_string(num_sims)}/ci_T{N_string(T)}_l{batch_len}_task{task_id}.pkl'
    else:
        filename = f'results_ncle/nle_summaries_N{N_string(num_sims)}/ci_T{N_string(T)}_l{batch_len}.pkl'
        
    with open(filename, 'wb') as f:
        pickle.dump(ci, f)
