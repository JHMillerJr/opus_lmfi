#> name: mcmc.py
#> author: John Miller Jr
#> descrp: mcmc function for Opus (7D project)
#          fits population-level galaxy params

""" #> IMPORTS =======================
================================== """



#> standard imports
import os
import sys
import numpy as np
import scipy.stats as stats
from tqdm import tqdm

import corner
import matplotlib.pyplot as plt

#> covariance prior imports
import pymc as pm
from scipy.stats import invwishart
from scipy.stats import wishart

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
def prior_LKJ():

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


def prior():
    
    global bounds
    lb, ub = bounds
    
    #> drawing mean vector and standard deviations
    
    # sample std deviations
    mu = np.random.uniform(lb,ub,size=d)
    sigma = np.random.uniform((ub-lb)/1e3,(ub-lb)/1e1,size=d)
    
    #> random wishart covariance?
    # https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf
    #A = np.random.normal(size=(d,d))
    #S = A @ A.T

    S = invwishart.rvs(df=d+1, scale=np.identity(d))
    # S = wishart.rvs(df=d, scale=np.identity(d))

    #> convert to correlation matrix
    Dcorr = np.sqrt(np.diag(S)) # .reshape(d,1) # needed if wanting to use Dcorr @ Dcorr.T
    R = S / np.outer(Dcorr, Dcorr) # why not Dcorr @ Dcorr.T? (it is the same!)
    
    cov = np.diag(sigma) @ R @ np.diag(sigma)
    
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
def createObs(outFile='', num_samples=1000):

    #> drawing sample
    mu, cov = prior()
    sample = model(mu, cov, num_samples)

    #> saving if requested
    if outFile != '':
        with open(outFile, 'w') as F:
            F.write(f'{mu}\t{cov.flatten()}\n')
            for line in sample:
                F.write(f'{line}\n')
        
    return mu, cov, sample


#> returns obs sample (either creates or takes existing sample)
def obs(outFile_obs  = dataDir + 'wabc_bivariate_obs.txt', 
        outFile_mock = dataDir + 'wabc_bivariate_mock.txt',
        create=False, num_samples=1000, verbose=True):

    #> if wanting to create obs sample
    if create: 
        #> creating 'observed' sample
        obs_mu, obs_cov, obs_sample = createObs(outFile_obs, num_samples=num_samples)
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
def cornerPlot(truth, outFile_mock, num_post=1000):
    
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
def checkConverg(outFile_mock, num_post=100, 
                 resample_perc=(2/3), num_resamples=100,
                 verbose=True):
    
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
        
        #> getting pearson correlation coeff
        posterior[:,3] /= ( np.sqrt(posterior[:,2]) * np.sqrt(posterior[:,4]) )
    
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
    
#> a single ABC sample
def abc_single(args):
    
    #> unpacking args
    obs_sample, num_slices = args

    #> declarations
    np.random.seed(None)

    #> sampling
    mu, cov = prior()
    mock_sample = model(mu, cov, num_samples=len(obs_sample))
    sw_dist = metric(obs_sample, mock_sample, num_slices)

    cov = np.delete(cov.flatten(), 2)
    return np.hstack([mu, cov, sw_dist])


#> parallelization of ABC
def runPABC(num_samples=10000, create_obs=False, num_obs_samples=1000,
            outFile_obs=dataDir + 'wabc_bivariate_obs.txt', 
            outFile_mock=dataDir + 'wabc_bivariate_mock.txt'):

    #> getting observed sample
    _, _, _, obs_sample = obs(outFile_obs=outFile_obs,
                              outFile_mock=outFile_mock,
                              create=create_obs,
                              num_samples=num_obs_samples,
                              verbose=False)

    #> declaration
    num_slices = 10
    workers = max(1, mp.cpu_count() - 1)
    print(f'> Using {workers} workers.')

    #> preparing args
    args = [(obs_sample, num_slices)] * num_samples

    #> main ABC run
    with mp.Pool(processes=workers) as pool:
        with open(outFile_mock, 'ab') as f:
            for r in tqdm(pool.imap(abc_single, args), total=num_samples):
                np.savetxt(f, r.reshape(1, -1))

    print(f'> Saved {num_samples} samples to {outFile_mock}')



""" #> MAIN ==========================
================================== """

#> declaring multivariate bounds
global d, bounds
d = 2
bounds = (-10, 10)


#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> initializing obs sample
    truth, obs_mu, obs_cov, obs_sample = obs(create=False)
    
    #> outFiles
    outFile_obs  = dataDir + 'wabc_bivariate_obs.txt'
    outFile_mock = dataDir + 'wabc_bivariate_mock.txt'
    
    #> abc declarations
    num_samples = 100000
    runPABC(num_samples=num_samples, outFile_obs=outFile_obs, outFile_mock=outFile_mock)
    
    
    #> checking posterior
    num_post = 1000
    cornerPlot(truth, outFile_mock, num_post=num_post)
    
    #> checking convergence
    post_mu, post_cov, post_std = checkConverg(outFile_mock, num_post=num_post)
    
    #> getting posterior sample
    post_sample = model(post_mu, post_cov, len(obs_sample))
    
    #> plotting obs and post
    bivarPlot(obs_sample, post_sample)
    
    # end
# thank