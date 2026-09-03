#> name: exp1.py
#> author: John Miller Jr
#> descrp: experiment to test SW with comparisons

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import time
import glob
import numpy as np
import scipy.stats as stats
from copy import deepcopy
import ot

import matplotlib.pyplot as plt
from scipy.stats import truncnorm as tnorm
from tqdm import tqdm

import multiprocessing as mp

import generate
import params
import modules.parse as parse

#> checking to see if GPU is available
# if error.checkGPU(): import cupy as cp

#> declarations
path = './metrics/exp1/'
sys.path.append(os.path.dirname(__file__)) 

#> plotting DPI
dpi = 200

#> whether to use GPU or CPU
global gpu
gpu = False


""" #> PLOTTING POP PARAMS ===========
================================== """

#> plotting distribution of galaxy params
def plotParamDist(name, pnames, muv, stdv, lbv, ubv):
    
    #> initializing plot
    fig, axes = plt.subplots(1,2,figsize=(12,6))
    
    plt.suptitle(name, fontweight='bold', fontsize=20)
    
    #> iterating through each param
    for ax, pname, mu, std, lb, ub in zip(axes.flat, pnames, muv, stdv, lbv, ubv):
        
        #> plotting stuff
        ax.grid(ls=':', alpha=0.5)
        
        #> getting dist
        a = (lb - mu) / std
        b = (ub - mu) / std
        x = np.linspace(lb, ub, 1000)
        y = tnorm.pdf(x, loc=mu, scale=std, a=a, b=b)
        
        #> plotting!
        ax.plot(x, y)
        ax.fill_between(x,y, alpha=0.2)
        ax.axvline(mu)
        
        #> more plotting stuff
        ax.set_xlabel(pname, fontweight='bold', fontsize=12)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        
    plt.savefig(path+f'{name}.png', bbox_inches='tight', dpi=dpi)
    plt.show()
    
    return


""" #> CREATE OBS ====================
================================== """

#> creating observed populations
def obs(numGals=100, numSource_gal=1):
    
    #> globals
    global gpu
    
    #> other declarations
    zl = 0.5
    zs = 1.0
    factor = 1
    nph = 50 * factor
    pix_arc = 60 * factor
    
    #> grid declarations
    xgrid, ygrid = generate.grid(nph=nph)
    
    #> getting galaxy profiles
    galProfiles = generate.galProfiles(mult=[1], ex=False)
    
    #> generating galaxy populations
    redshifts = np.array([(zl, zs)] * numGals)
    zl = redshifts[:,0]
    zs = redshifts[:,1]
    pix_arc = [pix_arc] * numGals

    #> if on CPU or GPU
    if gpu: import deflectionGPU as deflection
    else: import deflectionCPU as deflection
    
    #> getting paramRanges
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'norm':  {'init': 5e-3, 'min': 0.0, 'max':0.02, 'fit': True},
            'theta': {'init':  0.0, 'min': -90,'max': 90, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if dic.get('m',None) == 1 and param in ['norm', 'theta']:
                    dic[param] = vals[param]
                    continue
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> the five different populations (mu values)
    muv = [[0.000, 0.0],
           [0.005, 0.0],
           [0.010, 0.0],
           [0.005, 45.],
           [0.010, 45.]]
    muv = np.array(muv)
    
    #> sets std based on % of parameter range
    std_range_perc = 10
    
    #> determining std and saving boundaries
    stdv, lbv, ubv, pnames = [], [], [], []
    for key in vals.keys():
        rng = vals[key]['max'] - vals[key]['min']
        stdv.append((rng * std_range_perc / 100))
        lbv.append(vals[key]['min'])
        ubv.append(vals[key]['max'])
        pnames.append(key)
        
    #> observables
    observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1']
        
    #> creating all observed populations
    names = []
    fileNames = []
    for i, mu in enumerate(muv):
        
        #> name of observed sample
        name = f'obs{i+1}'
        names.append(name)
        print(f'> Generating {name}')
        
        if False: #> plots the distributions
            plotParamDist(name, pnames, mu, stdv, lbv, ubv)
        
        #> batch profiles
        bprofiles = params.bprofiles(numGals, paramRanges, 
                                     zl=zl, zs=zs, pix_arc=pix_arc,
                                     verbose=False, cov=None, mu=mu,
                                     std_range_perc=std_range_perc)
        
        #> calculating deflection angles
        lenses = generate.genGalPop(redshifts, galProfiles, bprofiles=bprofiles)
        
        #> lensing sources
        images, sources = generate.genQuadPop(lenses, zs, 
                                              observables=observables, jims=5,
                                              pix_arc=pix_arc, verbose=False, gpu=gpu,
                                              numSource_gal=numSource_gal)
        
        #> saving quad population
        fileName = path+'obs/'+name
        np.save(fileName, images)
        fileNames.append(fileName)
        
        #> saving galaxy params
        gal = {'muv': mu, 'stdv': stdv, 'lbv': lbv, 'ubv': ubv, 'pnames': pnames}
        np.save(path+'obs/'+name+'-gal', gal)
    
    return fileNames


""" #> CREATE MOCK ====================
================================== """

#> creating observed populations
def mock(numGals=1000, numSource_gal=1):
    
    #> globals
    global gpu
    
    #> other declarations
    zl = 0.5
    zs = 1.0
    factor = 1
    nph = 50 * factor
    pix_arc = 60 * factor
    
    #> grid declarations
    xgrid, ygrid = generate.grid(nph=nph)
    
    #> getting galaxy profiles
    galProfiles = generate.galProfiles(mult=[1], ex=False)
    
    #> generating galaxy populations
    redshifts = np.array([(zl, zs)] * numGals)
    zl = redshifts[:,0]
    zs = redshifts[:,1]
    pix_arc = [pix_arc] * numGals

    #> if on CPU or GPU
    if gpu: import deflectionGPU as deflection
    else: import deflectionCPU as deflection
    
    #> getting paramRanges
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'norm':  {'init': 5e-3, 'min': 0.0, 'max':0.02, 'fit': True},
            'theta': {'init':  0.0, 'min': -90,'max': 90, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if dic.get('m',None) == 1 and param in ['norm', 'theta']:
                    dic[param] = vals[param]
                    continue
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
    
    #> sets std based on % of parameter range
    std_range_perc = 10
    
    #> determining std and saving boundaries
    stdv, lbv, ubv, pnames = [], [], [], []
    for key in vals.keys():
        rng = vals[key]['max'] - vals[key]['min']
        stdv.append((rng * std_range_perc / 100))
        lbv.append(vals[key]['min'])
        ubv.append(vals[key]['max'])
        pnames.append(key)
        
    #> sampling uniforming the mu vector
    mu = np.random.uniform(lbv, ubv)
        
    #> observables
    observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1']
    
    # plotParamDist('mock', pnames, mu, stdv, lbv, ubv)
    
    #> batch profiles
    bprofiles = params.bprofiles(numGals, paramRanges, 
                                 zl=zl, zs=zs, pix_arc=pix_arc,
                                 verbose=False, cov=None, mu=mu,
                                 std_range_perc=std_range_perc)
    
    #> calculating deflection angles
    lenses = generate.genGalPop(redshifts, galProfiles, bprofiles=bprofiles)
    
    #> lensing sources
    images, sources = generate.genQuadPop(lenses, zs, 
                                          observables=observables, jims=5,
                                          pix_arc=pix_arc, verbose=False, gpu=gpu,
                                          numSource_gal=numSource_gal)
    
    return images, mu, stdv


#> the single ABC run
def mock_single(args):
    
    obs_samples, num_slices = args
    
    ranges = np.array([180, 90, 180, 1, 1, 1])
    
    #> generatin a mock sample, comparing to all obs samples, saving
    mock_sample, mu, sigma = mock(numGals=1000, numSource_gal=1)
    all_dists=[]
    for j, obs_sample in enumerate(obs_samples):
        dist = ot.sliced_wasserstein_distance(obs_sample/ranges, mock_sample/ranges, n_projections=num_slices)
        all_dists.append(dist)
    all_dists = np.array(all_dists)
        
    return np.hstack([mu, sigma, all_dists])
    
    


""" #> EXP. 1 ========================
================================== """

#> experiment one
def exp1():
    
    createObs=False
    
    #> mock declarations
    numMocks = 10000
    mockFile = path+'mockSamples.txt'
    
    #> create observed population
    if createObs:
        obsFileNames = obs(numGals=10, numSource_gal=1)
        obsFileNames = [x+'.npy' for x in obsFileNames]
        F = open(mockFile, 'w'); F.close()
    else:
        obsFileNames = glob.glob(path+'obs/obs*.npy')
    
    #> loading in observed populations
    obs_samples = []
    for file in obsFileNames:
        if 'g' in file: continue
        data = np.load(file, allow_pickle=True)
        obs_samples.append(data)
        
    
    #> declaration
    num_slices = 10
    workers = max(1, mp.cpu_count() - 10)
    print(f'> Using {workers} workers.')

    #> preparing args
    args = [(obs_samples, num_slices)] * numMocks
    
    #> main ABC run
    with mp.Pool(processes=workers) as pool:
        with open(mockFile, 'ab') as f:
            for r in tqdm(pool.imap(mock_single, args), total=numMocks):
                np.savetxt(f, r.reshape(1, -1))
                
    print(f'> Saved {numMocks} samples to {mockFile}')
            
    return

""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    exp1()
    
    # end
# thank