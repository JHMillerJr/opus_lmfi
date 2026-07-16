#> name: mcmc.py
#> author: John Miller Jr
#> descrp: mcmc function for Opus (7D project)
#          fits population-level galaxy params

""" #> IMPORTS =======================
================================== """

import pymc as pm

#> standard imports
import os
import sys
import random
import numpy as np
import scipy.stats as stats
from scipy.stats import invwishart
from tqdm import tqdm

import corner
import matplotlib.pyplot as plt

#> wasserstein
from ot import sliced_wasserstein_distance as swd

#> multiprocessing
import multiprocessing as mp

#> modules
from modules.units import u; u=u()
import modules.error as error


#> file declarations
dataDir = './data/'


""" #> PRIOR =========================
================================== """

#> multivariate prior w/ inverse wishart distribution
def prior():
    
    global bounds
    lb, ub = bounds
    
    #> drawing mean vector and standard deviations
    mu = np.random.uniform(4.99,5.01,size=d)
    mu[0] *= -1
    mu = np.random.uniform(lb, ub ,size=d)
    sigma = np.random.uniform((ub-lb)/1e3,(ub-lb)/1e2,size=d)
    
    #> random wishart covariance?
    # https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf
    #A = np.random.normal(size=(d,d))
    #S = A @ A.T

    S = invwishart.rvs(df=d+1, scale=np.identity(d))
    # S = wishart.rvs(df=d, scale=np.identity(d))

    #> convert to correlation matrix
    Dcorr = np.sqrt(np.diag(S)) # .reshape(d,1) # needed if wanting to use Dcorr @ Dcorr.T
    R = S / np.outer(Dcorr, Dcorr) # why not Dcorr @ Dcorr.T? (it is the same!)
    
    #> getting lower triangle values
    idx = np.tril_indices(d, k=-1)
    r = R[idx]
    
    # cov = np.diag(sigma) @ R @ np.diag(sigma)
    
    return np.hstack([mu, sigma, r])


""" #> MODEL =========================
================================== """

#> draws samples from bivariate normal
def model(theta, num_samples):
    
    #> unpacking
    mu, sigma, r= unpack(theta)
    cov = toCov(theta)
    
    return stats.multivariate_normal.rvs(mean=mu, cov=cov, size=num_samples)

#> unpacks theta in mu, sigma, r
def unpack(theta):
    
    #> getting dimension
    N = len(theta)                              # num free params
    d = int(( -3 + np.sqrt( 9 + (8*N) ) ) / 2)  # num dims
    
    #> posterior mean vector
    mu = theta[:d]
    sigma = theta[d:2*d]
    r = theta[-N+len(mu)*2:]
    
    return mu, sigma, r

#> unpacks theta and converts to covariance matrix
def toCov(theta):
    
    #> unpacking
    mu, sigma, r = unpack(theta)
    
    #> creating correlation
    R = np.identity(len(mu))
    idx = np.tril_indices(len(mu), k=-1)
    R[idx] = r; R.T[idx] = r
    
    #> creating covariance
    cov = np.diag(sigma) @ R @ np.diag(sigma)
    
    return cov
    

""" #> SLICED WASSERSTEIN ============
================================== """

#> returns the SW distance between two samples
def metric(sample1, sample2, L):
    return swd(sample1, sample2, n_projections=L)


""" #> OBS SAMPLE ====================
================================== """

#> creates observed sample
def createObs(num_samples=1000):
    
    #> globals
    global outFile_obs

    #> drawing sample
    theta = prior()
    sample = model(theta, num_samples)

    #> saving if requested
    if outFile_obs != '':
        with open(outFile_obs, 'w') as F:
            F.write(f'{theta}\n')
            for line in sample:
                F.write(f'{line}\n')
        
    return theta, sample


#> returns obs sample (either creates or takes existing sample)
def obs(create=False, num_samples=1000, verbose=True):
    
    #> globals
    global outFile_obs, outFile_mock

    #> if wanting to create obs sample
    if create: 
        
        #> creating 'observed' sample
        theta, obs_sample = createObs(num_samples=num_samples)
        
        #> clearing mock outFile (new obs --> new mocks)
        F = open(outFile_mock, 'w'); F.close()
        
    else:
        
        #> loading obs data
        with open(outFile_obs, 'r') as F:
    
            #> reading first line
            theta = np.array(F.readline().replace('[','').replace(']','').split(' '))
            theta = theta[ theta != '' ]
            theta = [ float(x) for x in theta ]
            
            #> loading in sample
            obs_sample = []
            obs_sample_txt = F.read().split('\n')
            for line in obs_sample_txt:
                array = np.array(line.replace('[','').replace(']','').replace('  ',' ').split(' '))
                array = array[ array != '' ]
                if len(array) < 2: continue  
                obs_sample.append([float(x) for x in array])
    
    #> printing obs
    if verbose:
        obs_mu, obs_sigma, obs_r = unpack(theta)
        print(f'> Observed mu={obs_mu}')
        print(f'> Observed sigma={obs_sigma}')
        print(f'> Observed r={obs_r}')
        print()
    
    return theta, obs_sample



""" #> PLOTTING ======================
================================== """

#> plots a bivariate distirbution
def bivarPlot(obs_sample, post_sample):
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    obs_sample = np.array(obs_sample)
    post_sample = np.array(post_sample)
    
    #> declarations
    fs = 12
    
    #> initializing plot
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.grid(ls=':', alpha=0.3)
    ax.set_title('Observed Sample vs. Posterior Sample', fontweight='bold')
    ax.set_xlabel('Var 1', fontweight='bold', fontsize=fs)
    ax.set_ylabel('Var 2', fontweight='bold', fontsize=fs)
    
    #> plotting!
    sns.kdeplot(x=obs_sample[:,0], y=obs_sample[:,1], color='b', ax=ax)
    sns.kdeplot(x=post_sample[:,0], y=post_sample[:,1], color='r', ax=ax)
    
    #> labels
    ax.plot([], c='b', label='obs')
    ax.plot([], c='r', label='post')
    plt.legend()
    
    plt.show()
    
    return


#> plots the corner plot of obs & mock samples
def cornerPlot(truth, num_post=1000):
    
    #> loading text files
    mock_data = np.loadtxt(outFile_mock)
    
    print(f'> There are {len(mock_data)} samples!')
    
    #> labels
    labels = [r'$\mu_1$', r'$\mu_2$', 
              r'$\Sigma_{11}$', r'$\Sigma_{12}$', r'$\Sigma_{22}$']
    
    #> checking posterior
    posterior = mock_data[ np.argsort(mock_data[:,-1]) ][:num_post]
    
    # epsilon_perc = 0.10 # % of mock_data taken for posterior
    # threshold = np.percentile(mock_data[:,-1], 100-(epsilon_perc*100))
    # posterior = mock_data[ mock_data[:,-1] >= threshold ]
    
    print(f'> Posterior contains {len(posterior)} samples or {len(posterior)/len(mock_data)*100:.2f}% of data.')
    
    #> plotting posterior
    fig = corner.corner(posterior[:,:-1], labels=labels, 
                        truths=truth, truth_color='r')
    plt.show()
    
    return


""" #> CONVERGENCE TEST ==============
================================== """

#> checks to see if converged
def checkConverg(num_post=100, resample_perc=(2/3), 
                 num_resamples=100, verbose=True):
    
    #> loading data
    mock_data = np.loadtxt(outFile_mock)
    
    #> resampling
    medians = np.zeros(shape=(num_resamples, mock_data.shape[1]-1))
    for num in range(num_resamples):
        
        #> getting random subset
        index = np.random.choice(mock_data.shape[0],
                                 int(len(mock_data) * resample_perc))
        subset = mock_data[index]
        
        #> checking posterior
        posterior = subset[ np.argsort(subset[:,-1]) ][:num_post]
    
        #> collecting moments
        for i in range(len(mock_data[0]) - 1):
            medians[num][i] += np.median(posterior[:,i])
    # medians /= num_resamples
    
    #> getting moments
    medians_mean = np.mean(medians, axis=0)
    medians_std = np.std(medians, axis=0)
    
    #> getting dimension
    N = len(medians_mean)                       # num free params
    d = int(( -3 + np.sqrt( 9 + (8*N) ) ) / 2)  # num dims
    
    #> posterior mean vector
    post_mu = medians_mean[:d]
    
    #> posterior covariance matrix
    idx = np.tril_indices(d, k=0)
    post_cov = np.zeros((d,d))
    post_cov[idx] = medians_mean[d:]
    post_cov.T[idx] = post_cov[idx]
    
    #> printing obs
    if verbose:
        print(f'> Posterior mu={post_mu}')
        print(f'> Posterior cov={medians_mean[d:]}')
    
    return post_mu, post_cov, medians_std


""" #> MAIN MCMC FN ==================
================================== """

#> function to perturb theta in MC kernel
def g(theta):
    
    #> globals
    global all_ranges
    range_perc = 10 # perc of range for scale (std)
    
    #> getting primed values
    theta_prime = stats.norm.rvs(loc=theta, scale=all_ranges/range_perc)
    cov = toCov(theta_prime)
    
    #> iterating until found
    max_tries = 100
    while not is_pos_def(cov):
        
        #> getting primed values
        theta_prime = stats.norm.rvs(loc=theta, scale=all_ranges/range_perc)
        cov = toCov(theta_prime)
        
        #> checking max tries
        max_tries -= 1
        if max_tries == 0: error.phrase('> COULD NOT FIND POSITIVE SEMI-DEFINITE')
        
    return theta_prime

#> returns true if matrix is positive semi-definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


#> the mc kernel
def MCKernel(theta, dist, epsilon,
             num_slices, num_mock_samples,
             obs_sample):
    
    ###> MARKOV KERNEL <### (1.1)
    
    #> declarations
    num_hits_r = 2
    
    #(A)> first resample
    L = []
    max_tries = 100
    print()
    while len(L) < num_hits_r and max_tries > 0:
        
        #> perturbing, sampling, comparing
        theta_prime_i = g(theta)
        prime_sample_i = model(theta_prime_i, num_mock_samples)
        prime_dist_i = swd(X_s=obs_sample, X_t=prime_sample_i, n_projections=num_slices)
        
        #> saving if less than threshold
        print(prime_dist_i, epsilon)
        if prime_dist_i <= epsilon: L.append([theta_prime_i, prime_dist_i])
        
        #> checking max tries
        max_tries -= 1
        # if max_tries == 0: error.phrase('> MC KERNEL COULD NOT FIND HIT (1)!')
    K_prime = len(L)
    
    #(B)> sample from hits
    if len(L) == 0: return theta, dist
    theta_prime_L = random.choices(population=L, weights=None)[0]
    
    #(C)> second resample
    L = []
    max_tries = 100
    while len(L) < (num_hits_r-1):
        
        #> perturbing, sampling, comparing
        theta_i = g(theta_prime_L[0])
        sample_i = model(theta_i, num_mock_samples)
        dist_i = swd(X_s=obs_sample, X_t=sample_i, n_projections=num_slices)
        
        #> saving if less than threshold
        if dist_i <= epsilon: L.append([theta_i, dist_i])
        
        #> checking max tries
        max_tries -= 1
        # if max_tries == 0: error.phrase('> MC KERNEL COULD NOT FIND HIT (2)!')
    K = len(L)
    
    #> accept / reject
    prime_prob = ( theta_prime_L[1] / dist ) * ( K / (K_prime - 1) )
    # print(prime_prob)
    if np.random.uniform() <= prime_prob: 
        return theta_prime_L[0], theta_prime_L[1]  # accepted
    else:
        return theta, dist  # rejected

#> main smc function
def smc(num_parts, num_gens):
    
    #> declarations
    global d, bounds
    num_obs_samples = 100
        
    #> generating observed data
    obs_theta = prior()
    obs_sample = model(obs_theta, num_obs_samples) # size d x num_obs_samples
    
    #> mock data declarations
    num_mock_samples = 3
    
    #> wasserstein declarations
    num_slices = 100
    
    
    ###> FIRST STEP <### (t=0)
    
    #(A)> sample theta from pi, set weights
    theta0 = np.array([ prior() for _ in range(num_parts) ])
    weights0 = np.full(shape=len(theta0), fill_value=1/num_parts)
    
    #(B)> sample z from mu (model given theta) and compute distances
    mock_samples0 = [ model(theta, num_mock_samples) for theta in theta0 ] # size num_particles x d x num_obs_samples
    dists0 = np.array([ swd(X_s=obs_sample, X_t=sample, n_projections=num_slices) for sample in mock_samples0 ])
    
    #(C)> based on theta0 and dists0, compute next threshold epsilon_1 (this is the simple threshold)
    epsilon_t_1 = np.percentile(dists0, 50)
    
    # ARE WEIGHTS0 GOING TO BE USED?
    
    
    ###> SECOND STEP <### (repeated step; t >= 1)
    
    #(A)> calculate and normalize weights
    weights_t = np.where(dists0, dists0 <= epsilon_t_1, 1.0)
    weights_t /= np.sum(weights_t)
    
    #(B)> sample ancestor indices
    indices = np.arange(num_parts)
    drawn_indices = random.choices(population=indices, weights=weights_t, k=num_parts)
    ancestor_thetas = theta0[drawn_indices]
    ancestor_distances = dists0[drawn_indices]

    #(C)> rejuvenation: sample new points from markov kernel
    theta_k_t, dist_k_t = [], []
    for theta, dist in zip(ancestor_thetas, ancestor_distances):
        print(dist, epsilon_t_1)
        tprime, dprime = MCKernel(theta, dist, epsilon_t_1, num_slices, num_mock_samples, obs_sample)
        theta_k_t.append(tprime); dist_k_t.append(dprime)
    theta_k_t = np.array(theta_k_t); dist_k_t = np.array(dist_k_t)
    
    ###> BACK TO STEP 2 <###
    #(D)> based on theta_t and dists_t, compute next threshold epsilon_t+1
    epsilon_t_1 = np.percentile(dist_k_t, 50)
    
    sys.exit()
    
    return 


""" #> MAIN ==========================
================================== """

#> declaring multivariate bounds
global d, bounds, outFile_obs, outFile_mock, all_ranges
d = 2
bound = 10
bounds = (-bound, bound)
all_ranges = np.array([bound*2, bound*2, bound*2/1e3, bound*2/1e1, 2])

#> outFiles
outFile_obs  = dataDir + 'wabc_bivariate_obs_smc.txt'
outFile_mock = dataDir + 'wabc_bivariate_mock_smc.txt'


#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> ensure data directory exists
    os.makedirs(dataDir, exist_ok=True)
    
    #> observed sample
    num_samples = 1000
    obs_theta, obs_sample = obs(create=False, num_samples=num_samples)
    
    #> abc-smc parameters
    num_particles = 10      # num of particles per generation
    num_generations = 5      # num of generations
    # epsilon_schedule = np.linspace(2.0, 0.05, num_generations)  # decreasing tolerances
    
    smc(num_parts=num_particles, num_gens=num_generations)

    
    #> posterior sample
    # post_sample = model(post_mu, post_cov, len(obs_sample))
    
    #> plots
    # bivarPlot(obs_sample, post_sample)
    # cornerPlot(truth, num_post=min(1000, len(particles)))
    
    # print(f'> posterior mu: {post_mu}')
    # print(f'> posterior cov: {post_cov}')
    
    # end
# thank