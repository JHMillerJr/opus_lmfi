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
from copy import deepcopy

import corner
import matplotlib.pyplot as plt

#> covariance prior imports
# import pymc as pm
from scipy.stats import invwishart
from scipy.stats import wishart
from scipy.stats import truncnorm
from scipy.optimize import fmin_slsqp

#> wasserstein
from ot import sliced_wasserstein_distance as swd

#> multiprocessing
import multiprocessing as mp

#> modules
import params
import deflection
import lensing
from modules.units import u; u=u()

#> truncated multivariate normal sampler
# from https://github.com/brunzema/truncated-mvn-sampler/blob/main/minimax_tilting_sampler.py
from modules.truncated_mvn_sampler.minimax_tilting_sampler import TruncatedMVN


#> file declarations
dataDir = './data/'


""" #> PRIOR =========================
================================== """

#> multivariate prior w/ inverse wishart distribution
def prior():
    
    global bounds
    lb = bounds[:,0]
    ub = bounds[:,1]
    
    #> drawing mean vector and standard deviations
    mu = np.random.uniform(lb, ub ,size=d)
    sigma = np.random.uniform((ub-lb)/1e3,(ub-lb)/1e2,size=d)

    #> sampling from an inverse wishart disttribution
    S = invwishart.rvs(df=d+1, scale=np.identity(d))

    #> convert to correlation matrix
    Dcorr = np.sqrt(np.diag(S)) # .reshape(d,1) # needed if wanting to use Dcorr @ Dcorr.T
    R = S / np.outer(Dcorr, Dcorr) # why not Dcorr @ Dcorr.T? (it is the same!)
    
    #> getting lower triangle values
    idx = np.tril_indices(d, k=-1)
    r = R[idx]
    
    # cov = np.diag(sigma) @ R @ np.diag(sigma)
    
    return mu, sigma, r


""" #> MODEL =========================
================================== """

#> draws samples from bivariate normal
def model(mu, sigma, r, num_samples):
    
    #> globals
    global nph, pix_arc, w_range, xgrid, ygrid
    global numQuads_gal, observables
    
    #> declarations
    numGals = int(num_samples / numQuads_gal)
    
    #> creating correlation
    R = np.identity(len(mu))
    idx = np.tril_indices(len(mu), k=-1)
    R[idx] = r; R.T[idx] = r
    
    #> creating covariance
    cov = np.diag(sigma) @ R @ np.diag(sigma)
    
    #> sampling from distribution
    bprofiles = params.bprofiles(numgals=numGals, 
                                 ranges=params.paramRanges(), 
                                 verbose=False, cov=cov, mu=mu)
    
    #> getting deflection angles
    delx, dely, lamt = deflection.deflectionGPU(xgrid, ygrid, bprofiles)
    
    #> lensing    
    images = []
    for i in range(numGals):
        for j in range(numQuads_gal):
            images.append(lensing.pfimsFn(xgrid, ygrid, 
                            delx[i], dely[i], 
                            lamt[i], 
                            pix_arc, nph, 
                            observables, bprofiles[i]))

    return np.array(images)


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
    mu, sigma, r = prior()
    sample = model(mu, sigma, r, num_samples)

    #> saving if requested
    if outFile_obs != '':
        with open(outFile_obs, 'w') as F:
            F.write(f'{mu}\t{sigma}\t{r}\n')
            for line in sample:
                F.write(f'{line}\n')
        
    return mu, sigma, r, sample


#> returns obs sample (either creates or takes existing sample)
def obs(create=False, num_samples=1000, verbose=True):
    
    #> globals
    global outFile_obs, outFile_mock

    #> if wanting to create obs sample
    if create: 
        
        #> creating 'observed' sample
        obs_mu, obs_sigma, obs_r, obs_sample = createObs(num_samples=num_samples)
        
        #> clearing mock outFile (new obs --> new mocks)
        F = open(outFile_mock, 'w'); F.close()
        
    else:
        
        #> loading obs data
        with open(outFile_obs, 'r') as F:
    
            #> reading first line
            obs_mu, obs_sigma, obs_r = F.readline().split('\t')
            
            print(obs_mu, obs_sigma, obs_r)
    
            #> converting
            obs_mu = np.array(obs_mu.replace('[','').replace(']','').replace('  ',' ').split(' ')[0:])
            obs_mu = obs_mu[ obs_mu != '' ]
            obs_mu = np.array([float(x) for x in obs_mu])
            
            obs_sigma = np.array(obs_sigma.replace('[','').replace(']','').replace('  ',' ').split(' ')[0:])
            obs_sigma = obs_sigma[ obs_sigma != '' ]
            obs_sigma = np.array([float(x) for x in obs_sigma])
            
            obs_r = [float(obs_r.replace('[','').replace(']','').replace('\n',''))]
            
            #> loading in sample
            obs_sample = []
            obs_sample_txt = F.read().split('\n')
            for line in obs_sample_txt:
                array = np.array(line.replace('[','').replace(']','').replace('  ',' ').split(' '))
                array = array[ array != '' ]
                if len(array) < 2: continue  
                obs_sample.append([float(x) for x in array])
    
    #> saving truth to be compared later
    truth = np.hstack([obs_mu, obs_sigma, obs_r])
    
    #> printing obs
    if verbose:
        print(f'> Observed mu={obs_mu}')
        print(f'> Observed sigma={obs_sigma}')
        print(f'> Observed r={obs_r}')
        print()
    
    return truth, obs_mu, obs_sigma, obs_r, obs_sample


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
    
    #> globals
    global outFile_obs, outFile_mock
    
    #> loading text files
    mock_data = np.loadtxt(outFile_mock)
    
    print(f'> There are {len(mock_data)} samples!')
    
    #> labels
    labels = [r'$\mu_1$', r'$\mu_2$', 
              r'$\sigma_{1}$', r'$\sigma_{2}$', r'$r_{12}$']
    
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

#> covariance matrix --> correlation matrix
def covToCorr(cov):
    dummy = deepcopy(cov)
    for i in range(len(dummy)):
        for j in range(len(dummy)):
            dummy /= ( np.sqrt(cov[i][i]) * np.sqrt(cov[j][j]) )
    if not all(np.diag(dummy)==1): print('> Correlation not correct!')
    return dummy


#> truncated normal fit (see https://stackoverflow.com/questions/53125437/fitting-data-using-scipy-truncnorm)
def tnorm(p, r, xa, xb):
    return truncnorm.nnlf(p, r) # negative log likelihood fn
def constraint(p, r, xa, xb):
    a, b, loc, scale = p
    return np.array([a*scale + loc - xa, b*scale + loc - xb])

#> checks to see if converged
def checkConverg(num_post=1000, resample_perc=(2/3), 
                 num_resamples=100, verbose=True, fit=False):
    
    #> globals
    global outFile_mock, bounds
    ub, lb = bounds
    
    #> loading data
    mock_data = np.loadtxt(outFile_mock)
    
    #> resampling
    medians = np.zeros(shape=(num_resamples, mock_data.shape[1]-1))
    stds = np.zeros(shape=(num_resamples, mock_data.shape[1]-1))
    tgauss_means = np.zeros(shape=(num_resamples, mock_data.shape[1]-1))
    tgauss_stds = np.zeros(shape=(num_resamples, mock_data.shape[1]-1))
    for num in range(num_resamples):
        
        #> getting random subset
        index = np.random.choice(mock_data.shape[0],
                                 int(len(mock_data) * resample_perc))
        subset = mock_data[index]
        
        #> checking posterior
        posterior = subset[ np.argsort(subset[:,-1]) ][:num_post]
    
        #> collecting moments
        tg_means, tg_stds = [], []
        for i in range(len(mock_data[0]) - 1):
            
            #> saving median
            data = posterior[:,i]
            medians[num][i] += np.median(data)
            stds[num][i] += np.std(data)

            if fit:
                #> declarations
                loc = np.median(data)
                scale = np.std(data)
                
                #> fitting truncnorm
                par = truncnorm.fit(data,
                                    f0=(np.min(data)-loc)/scale, 
                                    f1=(np.max(data)-loc)/scale,
                                    method='MM')
                tg_means.append(par[2])
                tg_stds.append(par[3])
        if fit: 
            tgauss_means[num] += tg_means
            tgauss_stds[num] += tg_stds
    
    #> getting moments
    if fit:
        medians_mean = np.median(tgauss_means, axis=0)
        medians_std = np.std(tgauss_means, axis=0)
        stds_mean = np.median(tgauss_stds, axis=0)
    else:
        medians_mean = np.median(medians, axis=0)
        medians_std = np.std(medians, axis=0)
        stds_mean = np.median(stds, axis=0)
    
    #> getting dimension
    N = len(medians_mean)                       # num free params
    d = int(( -3 + np.sqrt( 9 + (8*N) ) ) / 2)  # num dims
    
    #> posterior mean vector
    post_mu = medians_mean[:d]
    post_sigma = medians_mean[d:2*d]
    post_r = medians_mean[-N+len(post_mu)*2:]
    
    #> printing obs
    if verbose:
        print(f'> Posterior mu={post_mu}')
        print(f'> Posterior sigma={post_sigma}')
        print(f'> Posterior r={post_r}')
    
    return post_mu, post_sigma, post_r, medians_std


""" #> MAIN MCMC FN ==================
================================== """
    
#> a single ABC sample
def abc_single(args):
    
    #> unpacking args
    obs_sample, num_slices = args

    #> declarations
    np.random.seed(None)

    #> sampling
    mu, sigma, r = prior()
    mock_sample = model(mu, sigma, r, num_samples=len(obs_sample))
    sw_dist = metric(obs_sample, mock_sample, num_slices)

    return np.hstack([mu, sigma, r, sw_dist])


#> parallelization of ABC
def runPABC(num_samples=10000, create_obs=False, num_obs_samples=1000):
    
    #> globals
    global outFile_mock

    #> getting observed sample
    _, _, _, _, obs_sample = obs(create=create_obs,
                                 num_samples=num_obs_samples,
                                 verbose=False)

    #> declaration
    num_slices = 5
    workers = max(1, mp.cpu_count() - 1)
    print(f'> Using {workers} workers.')

    #> preparing args
    num_samples = int(num_samples)
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
global d, bounds, outFile_obs, outFile_mock
d = 2

#> outFiles
outFile_obs  = dataDir + 'wabc_bivariate_obs_sim.txt'
outFile_mock = dataDir + 'wabc_bivariate_mock_sim.txt'


global nph, pix_arc, w_range, xgrid, ygrid

#> grid declarations
nph = 50
pix_arc = 60 # default in bprofiles
w_range = np.arange(-nph, nph, 1, dtype=float)
xgrid, ygrid = np.meshgrid(w_range, w_range)


global numQuads_gal, observables
observables = ['dt23', 't23']
numQuads_gal = 1


global paramRanges
_, varied_params, _, _ = params.pruneParams()

bounds, pnames = [], []
for param in varied_params:
    pnames.append([param[0], param[2]])
    bounds.append([param[-1]['min'], param[-1]['max']])
bounds = np.array(bounds)


#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> initializing obs sample
    num_samples = 100
    truth, obs_mu, obs_sigma, obs_r, obs_sample = obs(create=False, num_samples=num_samples)
    
    # # # # ADD CHECK FOR EMPTY MOCKS AND OBS 
    
    #> abc declarations
    num_samples = 1000
    # runPABC(num_samples=num_samples)
    
    #> checking posterior
    num_post = 100
    cornerPlot(truth, num_post=num_post)
    
    #> checking convergence
    post_mu, post_sigma, post_r, post_std = checkConverg(num_post=num_post)
    
    #> getting posterior sample
    post_sample = model(post_mu, post_sigma, post_r, len(obs_sample))
    
    #> plotting obs and post
    bivarPlot(obs_sample, post_sample)

    # end
# thank