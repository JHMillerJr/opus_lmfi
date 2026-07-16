# > name: mcmc.py
# > author: John Miller Jr
# > descrp: mcmc function for Opus (7D project)
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
from scipy.stats import normaltest
from scipy.spatial.distance import jensenshannon

#> wasserstein
from ot import sliced_wasserstein_distance as swd

#> multiprocessing
import multiprocessing as mp

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

#> modules
from modules.units import u
u = u()


#> file declarations
dataDir = './metrics/w-data/'


""" #> PRIOR =========================
================================== """

#> number of variables for a given number of dimensions
def n_vars(x):
    return int(x*(x+3)/2)
    

#> multivariate prior w/ inverse wishart distribution
def prior(**kwargs):

    global bounds, d
    lb, ub = bounds
    
    #> sampling from posterior
    posterior_sampling = kwargs.get('posterior_sampling', [False] * n_vars(d) )

    #> drawing mean vector and standard deviations
    mu = np.random.uniform(4.99, 5.01, size=d)
    mu[0] *= -1
    mu = np.random.uniform(lb, ub, size=d)
    sigma = np.random.uniform((ub-lb)/1e3, (ub-lb)/1e2, size=d)
    
    #> sampling from posteior if converged
    for i, converged in enumerate(posterior_sampling):
        if converged:
            
            #> loading posterior and sampling
            posterior = np.load(dataDir + f'post_{i}.npy')
            sample = np.random.choice(posterior, size=1, replace=True)
            
            #> replacing
            if i == 0: mu[0] = sample
            if i == 1: mu[1] = sample
            if i == 2: sigma[0] = sample
            if i == 3: sigma[1] = sample
            

    #> random wishart covariance?
    # https://www.math.wustl.edu/~sawyer/hmhandouts/Wishart.pdf
    # A = np.random.normal(size=(d,d))
    # S = A @ A.T

    S = invwishart.rvs(df=d+1, scale=np.identity(d))
    # S = wishart.rvs(df=d, scale=np.identity(d))

    #> convert to correlation matrix
    # .reshape(d,1) # needed if wanting to use Dcorr @ Dcorr.T
    Dcorr = np.sqrt(np.diag(S))
    # why not Dcorr @ Dcorr.T? (it is the same!)
    R = S / np.outer(Dcorr, Dcorr)

    #> getting lower triangle values
    idx = np.tril_indices(d, k=-1)
    r = R[idx]

    # cov = np.diag(sigma) @ R @ np.diag(sigma)

    return mu, sigma, r


""" #> MODEL =========================
================================== """

#> draws samples from bivariate normal
def model(mu, sigma, r, num_samples):

    #> creating correlation
    R = np.identity(len(mu))
    idx = np.tril_indices(len(mu), k=-1)
    R[idx] = r
    R.T[idx] = r

    #> creating covariance
    cov = np.diag(sigma) @ R @ np.diag(sigma)

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
    global outFile_obs, outFile_mock, convergFile

    #> if wanting to create obs sample
    if create:

        #> creating 'observed' sample
        obs_mu, obs_sigma, obs_r, obs_sample = createObs(
            num_samples=num_samples)

        #> clearing mock outFile (new obs --> new mocks)
        F = open(outFile_mock, 'w')
        F.close()
        F = open(convergFile, 'w')
        F.close()

    else:

        #> loading obs data
        with open(outFile_obs, 'r') as F:

            #> reading first line
            obs_mu, obs_sigma, obs_r = F.readline().split('\t')

            print(obs_mu, obs_sigma, obs_r)

            #> converting
            obs_mu = np.array(obs_mu.replace('[', '').replace(
                ']', '').replace('  ', ' ').split(' ')[0:])
            obs_mu = obs_mu[obs_mu != '']
            obs_mu = np.array([float(x) for x in obs_mu])

            obs_sigma = np.array(obs_sigma.replace(
                '[', '').replace(']', '').replace('  ', ' ').split(' ')[0:])
            obs_sigma = obs_sigma[obs_sigma != '']
            obs_sigma = np.array([float(x) for x in obs_sigma])

            obs_r = [
                float(obs_r.replace('[', '').replace(']', '').replace('\n', ''))]

            #> loading in sample
            obs_sample = []
            obs_sample_txt = F.read().split('\n')
            for line in obs_sample_txt:
                array = np.array(line.replace('[', '').replace(
                    ']', '').replace('  ', ' ').split(' '))
                array = array[array != '']
                if len(array) < 2:
                    continue
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
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.grid(ls=':', alpha=0.3)
    ax.set_title('Observed Sample vs. Posterior Sample', fontweight='bold')
    ax.set_xlabel('Var 1', fontweight='bold', fontsize=fs)
    ax.set_ylabel('Var 2', fontweight='bold', fontsize=fs)

    #> plotting!
    sns.kdeplot(x=obs_sample[:, 0], y=obs_sample[:, 1], color='b', ax=ax)
    sns.kdeplot(x=post_sample[:, 0], y=post_sample[:, 1], color='r', ax=ax)

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
    posterior = mock_data[np.argsort(mock_data[:, -1])][:num_post]

    # epsilon_perc = 0.10 # % of mock_data taken for posterior
    # threshold = np.percentile(mock_data[:,-1], 100-(epsilon_perc*100))
    # posterior = mock_data[ mock_data[:,-1] >= threshold ]

    print(
        f'> Posterior contains {len(posterior)} samples or {len(posterior)/len(mock_data)*100:.2f}% of data.')

    #> plotting posterior
    fig = corner.corner(posterior[:, :-1], labels=labels,
                        truths=truth, truth_color='r',
                        show_titles=True, title_fmt='.3f')
    plt.show()

    return


""" #> CHECKING POSTERIOR ============
================================== """

#> covariance matrix --> correlation matrix
def covToCorr(cov):
    dummy = deepcopy(cov)
    for i in range(len(dummy)):
        for j in range(len(dummy)):
            dummy /= (np.sqrt(cov[i][i]) * np.sqrt(cov[j][j]))
    if not all(np.diag(dummy) == 1):
        print('> Correlation not correct!')
    return dummy


#> truncated normal fit (see https://stackoverflow.com/questions/53125437/fitting-data-using-scipy-truncnorm)
def tnorm(p, r, xa, xb):
    return truncnorm.nnlf(p, r)  # negative log likelihood fn

def constraint(p, r, xa, xb):
    a, b, loc, scale = p
    return np.array([a*scale + loc - xa, b*scale + loc - xb])

#> checking posterior
def post(num_post=1000, resample_perc=(2/3),
         num_resamples=100, verbose=True, fit=False):

    #> globals
    global outFile_mock, bounds
    lb, ub = bounds

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
        posterior = subset[np.argsort(subset[:, -1])][:num_post]

        #> collecting moments
        tg_means, tg_stds = [], []
        for i in range(len(mock_data[0]) - 1):

            # > saving median
            data = posterior[:, i]
            medians[num][i] += np.median(data)
            stds[num][i] += np.std(data)

            if fit:
                # > declarations
                loc = np.median(data)
                scale = np.std(data)

                # > fitting truncnorm
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
    d = int((-3 + np.sqrt(9 + (8*N))) / 2)  # num dims

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



#> checking prior draws from mock samples
def checkPrior(plot=False):
    
    #> loading data
    mock_data = np.loadtxt(outFile_mock)
    
    #> if plotting
    if plot: cornerPlot(truth, num_post=len(mock_data))
    
    return mock_data


""" #> CONVERGENCE ===================
================================== """

#> prior to posterior jensenshannon distance
def js_post_vs_prior(post, prior, lb, ub, bins=50):

    #> shared bins
    edges = np.linspace(lb, ub, bins + 1)

    #> posterior histogram
    p_hist, _ = np.histogram(post, bins=edges, density=False)

    #> prior histogram
    q_hist, _ = np.histogram(prior, bins=edges, density=False)

    #> getting prob vectors
    p_hist = p_hist.astype(float)
    q_hist = q_hist.astype(float)

    #> avoiding divide by zero
    if p_hist.sum() == 0 or q_hist.sum() == 0:
        return np.nan

    #> normalizing
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()

    return jensenshannon(p_hist, q_hist, base=2)


#> checking if converged
def checkConverg(num_post=1000, resample_perc=(2/3),
                 num_resamples=100, verbose=False, threshhold=0.6, **kwargs):

    #> globals
    global outFile_mock, d

    #> loading data
    mock_data = np.loadtxt(outFile_mock)
    
    #> getting bounds
    lb, ub = [], []
    for i in range(len(mock_data[0])-1):
        lb.append(min(mock_data[:,i]))
        ub.append(max(mock_data[:,i]))
    # print(lb,ub)
    
    #> setting threshholds
    if kwargs.get('threshholds', None) is None:
        threshholds = [threshhold] * n_vars(d)
    else:
        threshholds = kwargs.get('threshholds', None)
    

    #> resampling
    js_dists = []
    for num in range(num_resamples):

        #> getting random subset
        index = np.random.choice(mock_data.shape[0],
                                 int(len(mock_data) * resample_perc),
                                 replace=False)
        subset = mock_data[index]

        #> checking posterior
        posterior = subset[np.argsort(subset[:, -1])][:num_post]

        #> collecting moments
        dummy = []
        for i in range(len(mock_data[0]) - 1):

            #> saving median
            data = posterior[:, i]
            
            #> testing normal
            # stat, p = normaltest(data)
            dist = js_post_vs_prior(data, mock_data[:,i], 
                                    lb[i], ub[i])
            dummy.append([dist])
        js_dists.append(dummy)
    js_dists = np.array(js_dists)
    average_dists = np.mean(js_dists, axis=0)
    
    #> if wanting to print js distances
    if verbose:
        print(average_dists)
    
    #> checking convergence w/ treshhold
    if_converged = []
    for dist, thresh in zip(average_dists, threshholds):
        if_converged.append(dist >= thresh)
    if_converged = np.array(if_converged)
        
    return average_dists, if_converged


""" #> MAIN MCMC FN ==================
================================== """

#> a single ABC sample
def abc_single(args):

    # > unpacking args
    obs_sample, num_slices, posterior_sampling = args

    # > declarations
    np.random.seed(None)

    # > sampling
    mu, sigma, r = prior(posterior_sampling=posterior_sampling)
    mock_sample = model(mu, sigma, r, num_samples=len(obs_sample))
    sw_dist = metric(obs_sample, mock_sample, num_slices)

    return np.hstack([mu, sigma, r, sw_dist])


#> parallelization of ABC
def runPABC(num_samples=10000, create_obs=False, num_obs_samples=1000,
            convergence_sampling_rate=1000, threshhold=0.6, num_post=1000,
            threshhold_increment=0.05, **kwargs):

    #> globals
    global outFile_mock, d, convergFile

    #> getting observed sample
    _, _, _, _, obs_sample = obs(create=create_obs,
                                 num_samples=num_obs_samples,
                                 verbose=False)
    
    #> declaration
    num_slices = 5
    workers = max(1, mp.cpu_count() - 1)
    print(f'> Using {workers} workers.')
    convergence_sampling_rate = int(convergence_sampling_rate)
    
    #> if saving convergence info
    saveConverg = True

    #> preparing args
    num_samples = int(num_samples)
    posterior_sampling = [False] * n_vars(d)
    args = [(obs_sample, num_slices, posterior_sampling)] * convergence_sampling_rate
    
    #> setting up threshholds
    if kwargs.get('threshholds', None) is None:
        threshholds = [threshhold] * n_vars(d)
    else:
        threshholds = kwargs.get('threshholds')

    #> main ABC run
    when_converged = np.full(shape=(n_vars(d),), fill_value=np.inf)
    with mp.Pool(processes=workers) as pool:
        with open(outFile_mock, 'ab') as f:
            
            #> number of convergence checks
            for i in range(max(1, int(num_samples/convergence_sampling_rate))):
                
                print(f'> {i}')
                print(threshholds)
                
                #> sampling for a total of the convergence sampling rate
                for r in tqdm(pool.imap(abc_single, args), total=convergence_sampling_rate):
                    np.savetxt(f, r.reshape(1, -1))
                    
                #> checking to see if converged
                js_dists, if_converged = checkConverg(num_post=num_post, verbose=True,
                                                      threshholds=threshholds)
                args = [(obs_sample, num_slices, if_converged)] * convergence_sampling_rate
                
                #> if converged, save posterior
                for j, converged in enumerate(if_converged):
                    if converged: 
                        
                        #> saving when param converged
                        when_converged[j] = min(i*convergence_sampling_rate, 
                                                when_converged[j])
                        
                        #> increasing threshhold
                        threshholds[j] += threshhold_increment
                        
                        #> saving posterior at time of convergence
                        if when_converged[j] == i*convergence_sampling_rate:
                            
                            #> loading all mock data & getting posteriors
                            mock_data = np.loadtxt(outFile_mock)
                            posterior = mock_data[np.argsort(mock_data[:, -1])][:num_post][:,j]
                            np.save(dataDir+f'post_{j}', posterior)
                            print(f'> Saving var {j} posterior @ {(i+1)*convergence_sampling_rate} samples')
                
                #> saving info
                if saveConverg:
                    info = np.hstack([js_dists.reshape(1,n_vars(d)), 
                                      if_converged.reshape(1,n_vars(d)), 
                                      np.array(threshholds).reshape(1,n_vars(d))])
                    F = open(convergFile, 'a')
                    np.savetxt(F, info)
                    F.close()
                
    print(when_converged)
                    

    print(f'> Saved {num_samples} samples to {outFile_mock}')


""" #> MAIN ==========================
================================== """

#> declaring multivariate bounds
global d, bounds, outFile_obs, outFile_mock, convergFile
d = 2
bounds = (-10, 10)

#> outFiles
outFile_obs = dataDir + 'wabc_bivariate_obs.txt'
outFile_mock = dataDir + 'wabc_bivariate_mock.txt'
convergFile = dataDir + 'w_abc_converg_info.txt'


#> main function
if __name__ == '__main__':

    #> name
    print('> '+os.path.basename(__file__))

    #> initializing obs sample
    num_samples = 1000
    truth, obs_mu, obs_sigma, obs_r, obs_sample = obs(
        create=False, num_samples=num_samples)

    # # # # ADD CHECK FOR EMPTY MOCKS AND OBS
    
    #> threshhold for convergence
    convergence_threshhold = 0.8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    convergence_sampling_rate = 1e6
    threshholds = None

    #> abc declarations
    num_samples = 1e6
    #runPABC(num_samples=num_samples, 
    #        convergence_sampling_rate=convergence_sampling_rate,
    #        threshhold=convergence_threshhold,
    #        threshholds=threshholds)

    #> checking posterior
    num_post = 1000
    cornerPlot(truth, num_post=num_post)
    
    #> variable threshholds
    

    #> checking convergence
    js_dists, if_converged = checkConverg(num_post=num_post,
                                          threshhold=convergence_threshhold,
                                          threshholds=threshholds,
                                          verbose=True)
    print(if_converged)
    
    # checkPrior()

    sys.exit()

    #> getting posterior
    post_mu, post_sigma, post_r, post_std = post(num_post=num_post, fit=False)

    #> getting posterior sample
    post_sample = model(post_mu, post_sigma, post_r, len(obs_sample))

    #> plotting obs and post
    bivarPlot(obs_sample, post_sample)

    # end
# thank
