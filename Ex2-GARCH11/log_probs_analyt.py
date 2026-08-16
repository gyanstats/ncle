import os
import numpy as np
import pickle
from scipy.stats import norm
import time
import torch
import sys


# Simulator
def simulator(theta, T, burn_in=500):
    T_total = T + burn_in
    sigma_sq = np.zeros(T_total, dtype=np.float32)
    sigma_sq[0] = theta[1]/(1-theta[2]-theta[3])
    epsilon = np.zeros(T_total, dtype=np.float32)
    epsilon[0] = np.random.normal(theta[0], np.sqrt(sigma_sq[0]))
    for t in range(1, T_total):
        sigma_sq[t] = theta[1] + theta[2]*(epsilon[t-1]-theta[0])**2 + theta[3]*sigma_sq[t-1]
        epsilon[t] = np.random.normal(theta[0], np.sqrt(sigma_sq[t]))
    return epsilon[burn_in:]
    
    
# Observed data
T = int(sys.argv[1])
mu_0, omega_0, alpha_0, beta_0 = 0.5, 0.1, 0.1, 0.8 # true parameter values
theta_0 = [mu_0, omega_0, alpha_0, beta_0]

np.random.seed(0)
x_0 = simulator(theta_0, T=T) # generate x_0

l = int(sys.argv[2])
num_batches = T//l

param_list = ['mu', 'omega', 'alpha', 'beta']


def garch11_quasi_cll(theta, batches, num_batches):
    log_p_batches = []
    for b in range(num_batches):
        sigma_sq = np.zeros(l, dtype=np.float32)
        sigma_sq[0] = theta[1]/(1-theta[2]-theta[3])
        log_p_batch_b = norm.logpdf(batches[b][0], loc=theta[0], scale=np.sqrt(sigma_sq[0]))
        for t in range(1, l):
            sigma_sq[t] = theta[1] + theta[2]*(batches[b][t-1]-theta[0])**2 + theta[3]*sigma_sq[t-1]
            log_p_batch_b += norm.logpdf(batches[b][t], loc=theta[0], scale=np.sqrt(sigma_sq[t]))
        log_p_batches.append(log_p_batch_b)
        if (b+1) % 10_000 == 0:
            print(f'    batch: {b+1}/{num_batches}')
    return np.sum(log_p_batches)

def garch11_quasi_cll_probs(param, num_batches):
    if param == 'mu':
        n = 1000
        param_seq = np.linspace(0, 1, n+2)[1:-1]
    elif param == 'omega':
        n = 500
        param_seq = np.linspace(0, 0.3, n+2)[1:-1]
    elif param == 'alpha':
        n = 500
        param_seq = np.linspace(0, 1-theta_0[3], n+2)[1:-1]
    elif param == 'beta':
        n = 500
        param_seq = np.linspace(0.5, 1-theta_0[2], n+2)[1:-1]

    param_log_probs = np.zeros(n)
    batches = np.array_split(x_0, num_batches)
    for i in range(n):
        theta = theta_0.copy()
        theta[param_list.index(param)] = param_seq[i]
        param_log_probs[i] = garch11_quasi_cll(np.array(theta), batches, num_batches)
        if (i+1) % 50 == 0:
            print(f'  {param}: {i+1}/{n}')

    return param_log_probs
    

# Calculate log analytical CLs
log_probs_list_cl = {}
times_dict = {}

start = time.time()

for p in param_list:
    start_p = time.time()
    log_probs_list_cl[f"{p}"] = garch11_quasi_cll_probs(p, num_batches)
    times_dict[f"{p}"] = time.time() - start_p
    print(f'FINISHED PARAMETER {p}')
    print(f'Time: {times_dict[f"{p}"]}')

times_dict['total'] = time.time() - start
print(f'Total time: {times_dict["total"]}')

with open(f'log_probs_cl_l{l}.pkl', 'wb') as f:
    pickle.dump({'log_probs': log_probs_list_cl, 'times': times_dict}, f)
                                         

