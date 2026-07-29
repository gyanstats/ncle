import itertools
import numpy as np
import os
import pickle
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi
from sbi import utils
import sys
import time
import torch

# Parameters
T = 1000 # Sequence length
phi_0 = torch.tensor([0.8]) # True value

num_batches = int(sys.argv[1]) # adjusts automatically
batch_len = int( T/num_batches ) # Batch length
num_sims = 100_000 # Number of training simulations

# Prior
dim = 1
prior = utils.BoxUniform(low=-1 * torch.ones(dim), high=1 * torch.ones(dim))

# Simulator
def simulator(phi, T):
    X = np.zeros(T)
    X[0] = np.random.normal( 0, scale=np.sqrt(1/(1-phi**2)) )
    for t in range(1, T):
        X[t] = phi * X[t-1] + np.random.normal(0, 1)
    return X
    
    
# x_0
np.random.seed(0)
x_0 = simulator(phi=phi_0, T=T)
x_0 = np.float32(x_0)

# Set up sequence of phis for plotting later
n_ = 1000
phi_seq = np.linspace(-1, 1, n_+2)[1:-1]

# Infer phi and calculate log likelihood values for plotting
def ncle_train(num_batches, batch_len, num_sims):

    ''' Train NLE '''
    # Define prior
    dim = 1
    prior = utils.BoxUniform(low=-1 * torch.ones(dim), high=1 * torch.ones(dim))
    
    def simulator_N(phi: torch.Tensor, T: int, num_sims: int) -> torch.Tensor:
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

    # Learn likelihood estimator
    start_time = time.time()
    #simulator_sbi, prior = prepare_for_sbi(simulator_N, prior)
    #phi, x = simulate_for_sbi(simulator_sbi, proposal=prior, num_simulations=num_sims)
    # Generate N samples from prior and simulator
    phi = prior.sample((num_sims,)) # shape: (num_sims, 1)
    x = simulator_N(phi, batch_len, num_sims) # shape: (num_sims, batch_len)
    end_sim_time = time.time()
    
    inference = SNLE(prior=prior)
    likelihood_estimator = inference.append_simulations(phi, x).train()  # train NLE
    end_time = time.time()
    
    # Calculate elapsed times
    sim_time = end_sim_time - start_time
    train_time = end_time - end_sim_time
    nle_time = end_time - start_time

    ''' Calculate log likelihood given x_0 and a sequence of phi values phi_seq '''
    # Split x_0 into batches
    batches = np.array_split(x_0, num_batches)
    batches_mat = np.vstack(batches)

    # Calculate fused log p(x_0|phi) for a sequence of phi values
    nle_log_probs_batches = []
    nle_log_probs = np.zeros(n_)
    for i in range(n_):
        context = np.full((batches_mat.shape[0], 1), phi_seq[i], dtype=np.float32) # repeat the phi value num_batches times, since .log_prob() requires phi to have the same number of rows as batches_mat
        log_p_batches = likelihood_estimator.log_prob(batches_mat, context).detach().numpy()
        nle_log_probs_batches.append(log_p_batches) # gives vector log p(x_0b|phi), b=1,...,B
        nle_log_probs[i] = sum(log_p_batches) # gives fused log p(x_0|phi)

    return {"likelihood estimator": likelihood_estimator, "nle log probs batches": nle_log_probs_batches, "nle log probs": nle_log_probs,
            "nle time": nle_time, "simulation time": sim_time, "training time": train_time}


# Perform NCLE
training_results = ncle_train(num_batches, batch_len, num_sims)

# Save results to a file
task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
filename = f'ar1_{num_batches}batch_{task_id}.pkl'
with open(filename, 'wb') as f:
    pickle.dump(training_results, f)
    
    
