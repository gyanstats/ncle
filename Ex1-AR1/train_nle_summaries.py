import numpy as np
import os
import pickle
from sbi.inference import SNLE, simulate_for_sbi
from sbi import utils
import sys
import time
import torch

train_multiple = False

# Parameters
T = int(sys.argv[1]) # Sequence length
phi_0 = torch.tensor([0.8]) # True value

num_sims = int(sys.argv[2]) # Number of training simulations
batch_len = int(sys.argv[3]) # Batch length
num_batches = int( T/batch_len ) # Number of batches

# Prior
dim = 1
prior = utils.BoxUniform(low=-1 * torch.ones(dim), high=1 * torch.ones(dim))

def simulator(phi, T):
    # Extract scalar from tensor if needed
    phi_val = phi.item() if torch.is_tensor(phi) else float(phi)
    
    X = np.zeros(T)
    X[0] = np.random.normal(0, scale=np.sqrt(1/(1-phi_val**2)))
    for t in range(1, T):
        X[t] = phi_val * X[t-1] + np.random.normal(0, 1)
    return X
    
# x_0
np.random.seed(0)
x_0 = simulator(phi=phi_0, T=T)
x_0 = np.float32(x_0)

# For pickling at the end of the script
def N_string(N):
	if N >= 1_000_000: return f'{N//1_000_000}m'
	if N >= 1_000:     return f'{N//1_000}k'
	return str(N)

# Set up sequence of phis for plotting later
n_ = 1000
phi_seq = np.linspace(-1, 1, n_+2)[1:-1]

# Infer phi and calculate log likelihood values for plotting
def ncle_train(num_batches, batch_len, num_sims):

    ''' Train NLE '''
    # Define prior
    dim = 1
    prior = utils.BoxUniform(low=-0.99 * torch.ones(dim), high=0.99 * torch.ones(dim))
    
    def simulator_N(phi: torch.Tensor, T: int) -> torch.Tensor:
        """
        Simulate N AR(1) time series using torch only (vectorized).
        Args:
            phi (torch.Tensor): shape (N, 1) or (N,)
            T (int): time series length
        Returns:
            torch.Tensor: shape (N, T), each row is a simulated time series
        """
        phi = phi.view(-1) # Ensure shape is (N,)
        N = phi.shape[0]

        x = torch.empty(N, T)
        eps = torch.randn(N, T)

        std = torch.sqrt(1.0 / (1 - phi**2)) # shape: (N,)
        x[:, 0] = eps[:, 0] * std # shape (N,)

        for t in range(1, T):
            x[:, t] = phi * x[:, t - 1] + eps[:, t]

        return x

    def calculate_sufficient_stats(x: torch.Tensor) -> torch.Tensor:
        """
        Compute sufficient statistics for AR(1) model for multiple time series.
    
        Args:
            x (torch.Tensor): shape (N, T), N time series of length T
    
        Returns:
            torch.Tensor: shape (N, 2) containing [s1, s2] for each series
                s1 = sum_{t=2}^{T-1} x_t^2  
                s2  = sum_{t=2}^{T} x_{t-1} * x_t
        """
        N, T = x.shape
        
        # s1: sum of squares from t=2 to T-1
        s1 = torch.sum(x[:, 1:-1]**2, dim=1)  # shape: (N,)
    
        # s2: sum of products of adjacent observations
        s2 = torch.sum(x[:, :-1] * x[:, 1:], dim=1)  # shape: (N,)
    
        # Stack into (N, 2) tensor
        stats = torch.stack([s1, s2], dim=1)
    
        return stats
    
    
    # Learn likelihood estimator
    start_time = time.time()
    # Generate N samples from prior and simulator
    phi = prior.sample((num_sims,)) # shape: (num_sims, 1)
    x = simulator_N(phi, batch_len) # shape: (num_sims, batch_len)
    s = calculate_sufficient_stats(x) # shape: (num_sims, num_stats)
    end_sim_time = time.time()
    
    inference = SNLE(prior=prior)
    likelihood_estimator = inference.append_simulations(phi, s).train() # train NLE
    end_time = time.time()
    
    # Calculate elapsed times
    sim_time = end_sim_time - start_time
    train_time = end_time - end_sim_time
    nle_time = end_time - start_time

    ''' Calculate log likelihood given x_0 and a sequence of phi values phi_seq '''
    # 1. Split observed data into batches
    batches = np.array_split(x_0, num_batches)
    batches_mat = np.vstack(batches)  # (num_batches, batch_len)

    # 2. Compute sufficient statistics for each batch
    batches_torch = torch.tensor(batches_mat, dtype=torch.float32)
    s_0b = calculate_sufficient_stats(batches_torch).numpy()  # (num_batches, 3)

    # 3. Evaluate likelihood for each phi value
    nle_log_probs_batches = []
    nle_log_probs = np.zeros(n_)

    for i in range(n_):
        context = np.full((num_batches, 1), phi_seq[i], dtype=np.float32)
        log_p_batches = likelihood_estimator.log_prob(s_0b, context).detach().numpy()
        nle_log_probs_batches.append(log_p_batches)
        nle_log_probs[i] = sum(log_p_batches)

    return {"likelihood estimator": likelihood_estimator, "nle log probs batches": nle_log_probs_batches, "nle log probs": nle_log_probs,
            "nle time": nle_time, "simulation time": sim_time, "training time": train_time}


# Perform NLE
training_results = ncle_train(num_batches, batch_len, num_sims)

# Save results to a file
if train_multiple:
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    filename = f'results_ncle/nle_summaries_N{N_string(num_sims)}/train_l{batch_len}_{task_id}.pkl'
else:
    filename = f'results_ncle/nle_summaries_N{N_string(num_sims)}/train_l{batch_len}.pkl'
with open(filename, 'wb') as f:
    pickle.dump(training_results, f)
    
