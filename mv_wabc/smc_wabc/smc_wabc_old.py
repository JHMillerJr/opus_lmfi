#> name: mcmc.py
#> author: John Miller Jr
#> descrp: mcmc function for Opus (7D project)
#          fits population-level galaxy params

""" #> IMPORTS =======================
================================== """

import pymc as pm

#> standard imports
import os
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

import corner
import matplotlib.pyplot as plt

#> wasserstein
from ot import sliced_wasserstein_distance as swd

#> multiprocessing
import multiprocessing as mp

#> modules
from modules.units import u; u=u()


#> file declarations
dataDir = './data/'


""" #> PRIOR =========================
================================== """

#> returns a draw from the prior
def prior():

    #> unpacking global
    global bounds
    lb, ub = bounds
    
    #> drawing mean vector and standard deviations
    mu_model = pm.Uniform.dist(lower=lb, upper=ub, shape=d)
    sigma_model = pm.Uniform.dist(lower=(ub-lb)/1e3, upper=(ub-lb)/1e1, size=d)

    #> LKJ random correlation matrix R
    eta = 1.0     # uniform
    n = d         # num dims
    cov_model = pm.LKJCholeskyCov.dist(eta=eta, n=n, sd_dist=sigma_model)

    mu = pm.draw(mu_model)
    chol, corr, sigmas = pm.draw(cov_model)
    cov = chol @ chol.T # equiv to cov = np.diag(sigmas) @ corr @ np.diag(sigmas)

    return mu, cov


""" #> MODEL =========================
================================== """

#> draws samples from bivariate normal
def model(mu, cov, num_samples):
    return stats.multivariate_normal.rvs(mean=mu, cov=cov, size=num_samples)


""" #> SLICED WASSERSTEIN ============
================================== """

#> returns the SW distance between two samples
def metric(sample1, sample2, L):
    return swd(sample1, sample2, n_projections=L)


""" #> OBS SAMPLE ====================
================================== """

#> creates observed sample
def createObs(num_samples=1000):
    
    global outFile_obs

    #> drawing sample
    mu, cov = prior()
    sample = model(mu, cov, num_samples)

    #> saving if requested
    if outFile_obs != '':
        with open(outFile_obs, 'w') as F:
            F.write(f'{mu}\t{cov.flatten()}\n')
            for line in sample:
                F.write(f'{line}\n')
        
    return mu, cov, sample


#> returns obs sample (either creates or takes existing sample)
def obs(create=False, num_samples=1000, verbose=True):
    
    #> globals
    global outFile_obs, outFile_mock

    #> if wanting to create obs sample
    if create: 
        #> creating 'observed' sample
        obs_mu, obs_cov, obs_sample = createObs(num_samples=num_samples)
        obs_cov = np.delete(obs_cov.flatten(), 2)
        
        #> clearing mock outFile (new obs --> new mocks)
        F = open(outFile_mock, 'w'); F.close()
        
    else:
        
        #> loading obs data
        with open(outFile_obs, 'r') as F:
    
            #> reading first line
            obs_mu, obs_cov = F.readline().split('\t')
            
            print(obs_mu, obs_cov)
    
            #> converting
            obs_mu = np.array(obs_mu.replace('[','').replace(']','').replace('  ',' ').split(' ')[0:])
            obs_mu = np.array(obs_mu)
            obs_mu = obs_mu[ obs_mu != '' ]
            obs_mu = np.array([float(x) for x in obs_mu])
            
            obs_cov = obs_cov.replace('[','').replace(']','').replace('  ',' ').split(' ')[0:]
            obs_cov = np.array(obs_cov)
            obs_cov = obs_cov[ obs_cov != '' ]
            obs_cov = np.array([float(x) for x in obs_cov])
            obs_cov = np.delete(obs_cov, 2)
            
            #> loading in sample
            obs_sample = []
            obs_sample_txt = F.read().split('\n')
            for line in obs_sample_txt:
                array = np.array(line.replace('[','').replace(']','').split(' '))
                if len(array) != 2: continue
                obs_sample.append([float(x) for x in array])
    
    #> saving truth to be compared later
    truth = np.hstack([obs_mu, obs_cov])
    
    #> printing obs
    if verbose:
        print(f'> Observed mu={obs_mu}')
        print(f'> Observed cov={obs_cov}')
    
    return truth, obs_mu, obs_cov, obs_sample


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

#> single particle 
def smc_particle(args):

    #> unpacking params
    obs_sample, prev_particles, prev_weights, epsilon, perturb_cov = args
    
    #> declarations
    np.random.seed(None)
    d = len(obs_sample[0])

    if prev_particles is None:
        
        #> generation 0: sample from prior
        mu, cov = prior()
        
    else:
        
        #> resample previous particle
        idx = np.random.choice(len(prev_particles), p=prev_weights)
        particle = prev_particles[idx]
        
        #> perturb mu
        mu = particle[:d] + np.random.multivariate_normal(np.zeros(d), perturb_cov[:d, :d])
        
        #> perturb cov (flatten lower-tri)
        cov_flat = particle[d:-1] + np.random.multivariate_normal(np.zeros(len(particle[d:-1])), perturb_cov[d:-1, d-1])
        cov = np.zeros((d, d))
        idx_tril = np.tril_indices(d)
        cov[idx_tril] = cov_flat
        cov.T[idx_tril] = cov[idx_tril]

    #> simulate sample
    mock_sample = model(mu, cov, num_samples=len(obs_sample))
    dist = metric(obs_sample, mock_sample, L=100)

    return np.hstack([mu, cov.flatten()[:len(cov.flatten())-1], dist])


#> main smc function
def run_smc(num_parts, num_gens, epsilon_schedule,
            create_obs=False, num_obs_samples=1000):
    
    #> get observed sample
    _, _, _, obs_sample = obs(create=create_obs,
                              num_samples=num_obs_samples,
                              verbose=False)
    
    #> declarations
    particles, weights = None, None
    
    #> multiprocessing declarations
    workers = max(1, mp.cpu_count() - 1)
    print(f'> using {workers} workers')

    #> iterating through epsilons
    for t, epsilon in enumerate(epsilon_schedule):
        
        #> generation print statement
        print(f'> Generation {t+1}/{num_generations}, epsilon={epsilon:.4f}')
        
        #> perturb covariance (simple diagonal scaling)
        if particles is None:
            perturb_cov = None
        else:
            perturb_cov = np.cov(particles[:, :-1].T) + 1e-6*np.eye(particles.shape[1]-1)

        #> prepare args
        args = [(obs_sample, particles, weights, epsilon, perturb_cov)] * num_parts

        #> run particles in parallel
        with mp.Pool(processes=workers) as pool:
            results = list(tqdm(pool.imap(smc_particle, args), total=num_parts))

        results = np.array(results)
        
        #> accept particles within epsilon
        accepted = results[results[:, -1] <= epsilon]
        print(f'> accepted {len(accepted)} / {num_particles}')
        
        if len(accepted) == 0:
            raise RuntimeError('No particles accepted! Try increasing epsilon.')

        #> update weights
        if t == 0:
            weights = np.ones(len(accepted)) / len(accepted)
        else:
            #> simple uniform weights (replace with kernel density if desired)
            weights = np.ones(len(accepted)) / len(accepted)

        particles = accepted

        #> save to file
        with open(outFile_mock, 'ab') as f:
            np.savetxt(f, particles)

    print(f'> finished ABC-SMC, final particle count: {len(particles)}')
    return particles, weights


""" #> MAIN MCMC FN ==================
================================== """

#> main mcmc function
def mcmc():
    
    #> declarations
    global d, bounds
    num_obs_samples = 100
        
    #> generating observed data
    obs_mu, obs_cov = prior()
    thetaObs = obs_mu, obs_cov
    obs_sample = model(thetaObs, num_obs_samples) # size d x num_obs_samples
    
    #> mock data declarations
    num_particles = 4
    num_mock_samples = 3
    
    #> wasserstein declarations
    num_slices = 100
    
    
    ###> FIRST STEP <### (t=0)
    
    #> epsilon_0 = inf (all first samples are accepted)
    
    #(A)> sample theta from pi, set weights
    theta0 = [ prior() for _ in range(num_particles) ]
    weights0 = np.full(shape=len(theta0), fill_value=1/num_particles)
    
    #(B)> sample z from mu (model given theta) and compute distances
    mock_samples0 = [ model(theta, num_mock_samples) for theta in theta0 ] # size num_particles x d x num_obs_samples
    dists0 = [ swd(X_s=obs_sample, X_t=sample, n_projections=num_slices) for sample in mock_samples0 ]
    
    #(C)> based on theta0 and dists0, compute next threshold epsilon_1
    epsilon_t_1 = np.percentile(dists0, 50)
    
    # ARE WEIGHTS0 GOING TO BE USED?
    
    
    ###> SECOND STEP <### (repeated step; t >= 1)
    
    #(A)> calculate and normalize weights
    weights_t = np.where(dists0, dists0 <= epsilon_t_1, 1.0)
    weights_t /= np.sum(weights_t)
    
    #(B)> sample ancestor indices
    indices = np.arange(num_particles) + 1
    new_indices = random.choices(population=indices, weights=weights_t, k=num_particles)

    #(C)> rejuvenation: sample new points from markov kernel
    
    ###> MARKOV KERNEL <### (1.1)
    
    
    
    
    ###> BACK TO STEP 2 <###
    #(D)> based on theta_t and dists_t, compute next threshold epsilon_t+1
    
    sys.exit()
    
    return 


""" #> MAIN ==========================
================================== """

#> declaring multivariate bounds
global d, bounds, outFile_obs, outFile_mock
d = 2
bounds = (-10, 10)

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
    truth, obs_mu, obs_cov, obs_sample = obs(create=True, num_samples=num_samples)
    
    #> abc-smc parameters
    num_particles = 100      # num of particles per generation
    num_generations = 5      # num of generations
    epsilon_schedule = np.linspace(2.0, 0.05, num_generations)  # decreasing tolerances
    
    #> run ABC-SMC
    particles, weights = run_smc(num_parts=num_particles,
                                 num_gens=num_generations,
                                 epsilon_schedule=epsilon_schedule)
    
    #> extract posterior mean & cov for plotting
    d = len(obs_mu)
    post_mu = np.mean(particles[:, :d], axis=0)
    
    #> reconstruct cov from lower-triangular flattened part
    post_cov = np.zeros((d, d))
    idx_tril = np.tril_indices(d)
    cov_flat = np.mean(particles[:, d:-1], axis=0)
    post_cov[idx_tril] = cov_flat
    post_cov.T[idx_tril] = post_cov[idx_tril]
    
    #> posterior sample
    post_sample = model(post_mu, post_cov, len(obs_sample))
    
    #> plots
    bivarPlot(obs_sample, post_sample)
    cornerPlot(truth, num_post=min(1000, len(particles)))
    
    print(f'> posterior mu: {post_mu}')
    print(f'> posterior cov: {post_cov}')
    
    # end
# thank