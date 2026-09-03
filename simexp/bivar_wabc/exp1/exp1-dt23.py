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

#> checking to see if GPU is available
# if error.checkGPU(): import cupy as cp

#> declarations
path = './metrics/exp1/'
sys.path.append(os.path.dirname(__file__)) 
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import generate
import params
import modules.parse as parse

#> plotting DPI
dpi = 200

#> whether to use GPU or CPU
global gpu
gpu = False


""" #> MEMORY CALC ===================
================================== """

#> converts memory units
def bytesto(bytes, to, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    return bytes / (bsize ** a[to])

#> converts memory units
def tobytes(bytes, from_, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    r = float(bytes)
    return bytes / (bsize ** -a[from_])


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

    #> creating all observed populations
    names = []
    fileNames = []
    for i, mu in enumerate(muv):
        
        #> name of observed sample
        name = f'obs{i+1}-dt23'
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
    
    # plotParamDist('mock', pnames, mu, stdv, lbv, ubv)
    
    #> batch profiles
    bprofiles = params.bprofiles(numGals, paramRanges, 
                                 zl=zl, zs=zs, pix_arc=pix_arc,
                                 verbose=False, cov=None, mu=mu,
                                 std_range_perc=std_range_perc)
    
    #> calculating deflection angles
    lenses = generate.genGalPop(redshifts, galProfiles, bprofiles=bprofiles, gpu=gpu)
    
    #> lensing sources
    images, sources = generate.genQuadPop(lenses, zs, 
                                          observables=observables, jims=5,
                                          pix_arc=pix_arc, verbose=False, gpu=gpu,
                                          numSource_gal=numSource_gal)
    
    return images, mu, stdv


#> the single ABC run
def mock_single(args):
    
    #> generatin a mock sample, comparing to all obs samples, saving
    mock_sample, mu, sigma = mock(numGals=1000, numSource_gal=1)
    print('1')
    all_dists=[]
    for j, obs_sample in enumerate(obs_samples):
        print('2-j')
        dist = ot.sliced_wasserstein_distance(obs_sample/ranges, mock_sample/ranges, n_projections=num_slices)
        print(dist)
        all_dists.append(dist)
    all_dists = np.array(all_dists)
        
    return np.hstack([mu, sigma, all_dists])
    
    

""" #> EXP. 1 ========================
================================== """

#> experiment one
def exp1(numMocks):
    
    #> declaration
    num_slices = 10
    
    print(f'> There are {mp.cpu_count()} cores available!')
    
    #> parallelization declarationss
    # numCores = 128
    # gpuMemory = 48 # gb
    # workers =  min(numCores, int( tobytes(gpuMemory, 'g') / ( numGals * 800000 * 57.5 ) ))
    # workers = 9
    workers = max(1, mp.cpu_count() - 1)
    workers=4
    print(workers)
    
    print(f'> Using {workers} workers.')
    
    args = [None] * numMocks
    
    #> main ABC run
    with mp.Pool(processes=workers) as pool:
        with open(mockFile, 'ab') as f:
            for r in tqdm(pool.imap_unordered(mock_single, args), total=numMocks):
                np.savetxt(f, r.reshape(1, -1))

    #mp.set_start_method('spawn', force=True)
    #workers = 1
    #for r in tqdm(map(mock_single), total=numMocks):
    #    pass
                
    print(f'> Saved {numMocks} samples to {mockFile}')
            
    return


##> DECLARATIONS

global observables, ranges
observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1']
ranges = np.array([180, 90, 180, 1, 1, 1])

global num_slices
num_slices = 10

#> mock declarations
global mockFile
mockFile = path+'mockSamples.txt'

#> observed data
createObs=False

#> create observed population
if createObs:
    obsFileNames = obs(numGals=100, numSource_gal=1)
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
    
global OBS_SAMPLES
OBS_SAMPLES = obs_samples



""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    srt = time.time()
    numMocks=100
    exp1(numMocks)
    ttl = time.time() - srt
    print(f'> total time is {ttl/60} minutes')
    print(f'> ttl time/ mock = {ttl/numMocks} seconds')
    
    # end
# thank