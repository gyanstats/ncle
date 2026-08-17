import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from sbi.inference import SNLE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import likelihood_nn
import sys
import time
import torch
from torch.distributions.uniform import Uniform
import torch.distributions as dist

train_multiple = False

# Parameters
num_sims = int(sys.argv[1]) # Number of training simulations
batch_len = int(sys.argv[2])

# Prior over (mu, sigma2_uncond, persistence, alpha_frac)
class garch11_prior(dist.Distribution):
    def __init__(self):
        super().__init__()
        self.mu_dist        = dist.Uniform(-1, 1)
        self.sigma2_dist    = dist.Uniform(0, 2)      # unconditional variance
        self.persistence_dist = dist.Uniform(0, 1)    # alpha + beta
        self.alpha_frac_dist  = dist.Uniform(0, 1)    # alpha / persistence

    def sample(self, sample_shape=torch.Size()):
        mu          = self.mu_dist.sample(sample_shape)
        sigma2      = self.sigma2_dist.sample(sample_shape)
        persistence = self.persistence_dist.sample(sample_shape)
        alpha_frac  = self.alpha_frac_dist.sample(sample_shape)
        return torch.stack([mu, sigma2, persistence, alpha_frac], dim=-1)

    def log_prob(self, values):
        mu, sigma2, persistence, alpha_frac = (
            values[..., 0], values[..., 1], values[..., 2], values[..., 3]
        )
        valid = (
            (-1 <= mu) & (mu <= 1) &
            (0 <= sigma2) & (sigma2 <= 2) &
            (0 <= persistence) & (persistence <= 1) &
            (0 <= alpha_frac) & (alpha_frac <= 1)
        )
        # flat prior on rectangular support: log density = log(1/2 * 1/2 * 1 * 1) = -log(4)
        log_density = torch.full_like(mu, -torch.log(torch.tensor(4.0)))
        return torch.where(valid, log_density, torch.tensor(-float("inf")))


def unpack_reparam(thetas):
    """Convert reparameterised (mu, sigma2, persistence, alpha_frac) -> (mu, omega, alpha, beta)."""
    mu          = thetas[:, 0]
    sigma2      = thetas[:, 1]
    persistence = thetas[:, 2]
    alpha_frac  = thetas[:, 3]

    alpha = alpha_frac * persistence
    beta  = (1 - alpha_frac) * persistence
    omega = sigma2 * (1 - persistence)
    return mu, omega, alpha, beta


def simulator_N(thetas, T, num_sims, burn_in=500):
    mu, omega, alpha, beta = unpack_reparam(thetas)
    sigma2_uncond = thetas[:, 1]  # grab directly for initialisation

    T_total = T + burn_in
    x        = torch.zeros(num_sims, T_total)
    sigma_sq = torch.zeros(num_sims, T_total)

    sigma_sq[:, 0] = sigma2_uncond
    x[:, 0]        = torch.normal(mu, torch.sqrt(sigma_sq[:, 0]))

    for t in range(1, T_total):
        sigma_sq[:, t] = omega + alpha * (x[:, t-1] - mu)**2 + beta * sigma_sq[:, t-1]
        x[:, t]        = torch.normal(mu, torch.sqrt(sigma_sq[:, t]))

    return x[:, burn_in:]

# Infer theta
if __name__ == '__main__':
    
    # Define prior
    prior = garch11_prior()
        
    # Learn likelihood estimator
    start_time = time.time()
    # Generate N samples from prior and simulator
    theta = prior.sample((num_sims,)) # shape: (num_sims, 4)
    x = simulator_N(theta, T=batch_len, num_sims=num_sims) # shape: (num_sims, batch_len)
    end_sim_time = time.time()
    # Save training data
    training_data = {'theta': theta, 'x': x}

    inference = SNLE(prior=prior, device='cpu')

    likelihood_estimator = inference.append_simulations(theta, x).train()  # likelihood estimator
    end_time = time.time()
    
    # Calculate elapsed times
    sim_time = end_sim_time - start_time
    train_time = end_time - end_sim_time
    nle_time = end_time - start_time
    
    training_results = {"likelihood estimator": likelihood_estimator, "training_data": training_data,
                       "nle time": nle_time, "simulation time": sim_time, "training time": train_time}


    # Save results to a file
    if train_multiple:
        task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
        filename = f'results_ncle/nle_{num_sims}/train_l{batch_len}_{task_id}.pkl'
    else:
        filename = f'results_ncle/nle_{num_sims}/train_l{batch_len}.pkl'
        
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(training_results, f)
