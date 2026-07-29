import numpy as np
import os
import pickle
import sys
import time
import torch

# Simulator
def simulator(theta, T):
    sigma_sq = np.zeros(T, dtype=np.float32) # initialise
    sigma_sq[0] = theta[1]/(1-theta[2]-theta[3]) # set initial value of sigma squared to be ω/(1-α-β) (provided that α+β<1)
    epsilon = np.zeros(T, dtype=np.float32) # initialise
    epsilon[0] = np.random.normal(theta[0], np.sqrt(sigma_sq[0])) # simulate initial epsilon
    # Simulate GARCH(1,1) process
    for t in range(1, T):
        sigma_sq[t] = theta[1] + theta[2]*(epsilon[t-1]-theta[0])**2 + theta[3]*sigma_sq[t-1]
        epsilon[t] = np.random.normal(theta[0], np.sqrt(sigma_sq[t]))
    return epsilon

# True parameter values
mu_0, omega_0, alpha_0, beta_0 = 0.5, 0.1, 0.1, 0.8
theta_0 = [mu_0, omega_0, alpha_0, beta_0]
# True reparameterised values
sigma2_0 = omega_0 / (1 - alpha_0 - beta_0)
persistence_0 = alpha_0 + beta_0
alpha_frac_0 = alpha_0 / persistence_0

T = int(sys.argv[1]) # seq length
l = int(sys.argv[2]) # batch length
num_batches = T//l

# Observed data
np.random.seed(0)
x_0 = simulator(theta_0, T=T) # generate x_0
# Split x_0 into batches
batches_mat = np.vstack(np.array_split(x_0, num_batches)) # shape: (num_batches, batch_len)

param_list = ['mu', 'omega', 'alpha', 'beta']

with open(f'train_l{l}_1.pkl', 'rb') as f:
    nle_batches = pickle.load(f)
likelihood_estimator = nle_batches['likelihood estimator']
    
def calculate_log_probs(param, num_batches, likelihood_estimator):

    def cll(theta):
        context = np.full((batches_mat.shape[0], 4), theta, dtype=np.float32)
        return sum(likelihood_estimator.log_prob(batches_mat, context)).detach().numpy()

    param_configs = {
        'mu':         (1000, np.linspace(0, 1, 1002)[1:-1],
                       lambda v: [v, sigma2_0, persistence_0, alpha_frac_0]),
        'omega':      (500,  np.linspace(0, 0.3, 502)[1:-1],
                       lambda v: [mu_0, v/(1-persistence_0), persistence_0, alpha_frac_0]),
        'alpha': (500,  np.linspace(0, 1-beta_0, 502)[1:-1],
                  lambda v: [mu_0, sigma2_0 * (1 - persistence_0) / (1 - v - beta_0),
                             v + beta_0, v / (v + beta_0)]),
        'beta':  (500,  np.linspace(0.5, 1-alpha_0, 502)[1:-1],
                 lambda v: [mu_0, sigma2_0 * (1 - persistence_0) / (1 - alpha_0 - v),
                            alpha_0 + v, alpha_0 / (alpha_0 + v)]),
    }

    n, param_seq, theta_fn = param_configs[param]
    param_log_probs = np.zeros(n)
    for i in range(n):
        param_log_probs[i] = cll(torch.tensor(theta_fn(param_seq[i]), dtype=torch.float32))
        if (i+1) % 50 == 0:
            print(f'  {param}: {i+1}/{n}')

    return param_log_probs
    
# Calculate log analytical CLs
log_probs_list_ncl = {}
times_dict = {}

start = time.time()

for p in param_list:
    start_p = time.time()
    log_probs_list_ncl[f"{p}"] = calculate_log_probs(p, num_batches, likelihood_estimator)
    times_dict[f"{p}"] = time.time() - start_p
    print(f'FINISHED PARAMETER {p}')
    print(f'Time: {times_dict[f"{p}"]}')

times_dict['total'] = time.time() - start
print(f'Total time: {times_dict["total"]}')

with open(f'log_probs_ncl_l{l}.pkl', 'wb') as f:
    pickle.dump({'log_probs': log_probs_list_ncl, 'times': times_dict}, f)
                                         

