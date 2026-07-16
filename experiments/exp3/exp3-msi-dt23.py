#> name: exp3.py
#> author: John Miller Jr
#> descrp: experiment to test SW with comparisons

""" #> IMPORTS =======================
================================== """

#> current working directory
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

#> changing backends
os.environ["POT_BACKEND"] = "numpy"   # ← THIS is the key fix
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.modules["cupy"] = None
sys.modules["jax"] = None
sys.modules["torch"] = None
sys.modules["tensorflow"] = None

#> standard imports
import time
import glob
import numpy as np

# import matplotlib.pyplot as plt
from scipy.stats import truncnorm as tnorm
from tqdm import tqdm
from ot import sliced_wasserstein_distance as swd

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

#> declarations
dir_path = os.path.dirname(os.path.realpath(__file__)) + '/'

import generate
import params

#> whether to use GPU or CPU
global gpu
gpu = False

import multiprocessing as mp


""" #> MEMORY CALC ===================
================================== """

#> converts memory units
def bytesto(bytes, to, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    # r = float(bytes)
    return bytes / (bsize ** a[to])

#> converts memory units
def tobytes(bytes, from_, bsize=1024): 
    a = {'k' : 1, 'm': 2, 'g' : 3, 't' : 4, 'p' : 5, 'e' : 6 }
    # r = float(bytes)
    return bytes / (bsize ** -a[from_])


""" #> PLOTTING POP PARAMS ===========
================================== """

#> plotting distribution of galaxy params
def plotParamDist(name, pnames, muv, stdv, lbv, ubv):
    
    #> imports
    import matplotlib.pyplot as plt

    #> declarations
    dpi = 200
    
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
        
    plt.savefig(dir_path+f'{name}.png', bbox_inches='tight', dpi=dpi)
    plt.show()
    
    return


""" #> CREATE OBS ====================
================================== """

#> creating observed populations
def obs(numGals, numSource_gal, observables):
    
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
    
    #> getting paramRanges
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'norm':  {'init': 5e-3, 'min': 0.0, 'max':0.02, 'fit': True},
            'theta': {'init':  0.0, 'min': 0,'max': 90, 'fit': True}}
    
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
        fileName = dir_path+'obs/'+name
        np.save(fileName, images)
        fileNames.append(fileName)
        
        #> saving galaxy params
        gal = {'muv': mu, 'stdv': stdv, 'lbv': lbv, 'ubv': ubv, 'pnames': pnames}
        np.save(dir_path+'obs/'+name+'-gal', gal)
    
    return fileNames


""" #> CREATE MOCK ====================
================================== """

#> creating observed populations
def mock(numGals=1000, numSource_gal=1):
    
    global OBSERVABLES
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
    
    #> getting paramRanges
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'norm':  {'init': 5e-3, 'min': 0.0, 'max':0.02, 'fit': True},
            'theta': {'init':  0.0, 'min': 0,'max': 90, 'fit': True}}
    
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
                                          observables=OBSERVABLES, jims=5,
                                          pix_arc=pix_arc, verbose=False, gpu=gpu,
                                          numSource_gal=numSource_gal)
    
    return images, mu, stdv
    

#> single mock
def mock_single(_):
    
    #> declarations
    num_slices = 10
    numGals = 1000
    numSource_gal = 1
    
    #> generating mock sample
    mock_sample, mu, sigma = mock(numGals=numGals, numSource_gal=numSource_gal)
    
    #> comparing against all observed samples
    all_dists = []
    for obs_sample in OBS_SAMPLES:
        
        #> getting the ranges for each parameter
        ranges = np.ptp(obs_sample, axis=0)
        
        #> calculating the distance
        dist = swd(
            obs_sample / ranges,      # normalized obs
            mock_sample / ranges,     # normalized mock
            n_projections=num_slices) # num slices
        all_dists.append(dist)

    return np.hstack([mu, sigma, np.array(all_dists)])
    


""" #> EXP. 3 ========================
================================== """

#> experiment three
def exp3(args):
    
    #> unpacking
    numMocks, workers, mockFile, observables, createObs = args
    
    #> workers
    if workers is None: workers = 1
    if mp.cpu_count() < workers:
        import modules.error as error
        error.highlight(f'Workers(={workers}) > CPUs(={mp.cpu_count()})')
    print(f'> Using {workers} workers.')
    
    #> create observed population
    if createObs:
        _ = obs(numGals=100, numSource_gal=1, observables=observables)
        print(f'> Overwriting mockFile: {mockFile}')
        F = open(mockFile, 'w'); F.close()
    
    #> main ABC run
    with mp.Pool(processes=workers, initializer=init_worker, initargs=(observables,)) as pool:
        with open(mockFile, 'ab') as f:
            for r in tqdm(pool.imap_unordered(mock_single, range(numMocks)), total=numMocks):
                np.savetxt(f, r.reshape(1, -1))
                
    print(f'> Saved {numMocks} samples to {mockFile}')
            
    return


#> initializing workers
def init_worker(observables):
    
    #> setting globals
    global OBSERVABLES, OBS_SAMPLES
    OBSERVABLES = observables
    
    #> getting observed sample
    OBS_SAMPLES = []
    for file in glob.glob(dir_path+'obs/obs*dt23.npy'):
        if 'dt23' not in file or 'g' in file:
            continue
        OBS_SAMPLES.append(np.load(file, allow_pickle=True))
    
    #> resetting random seed (just in case)
    np.random.seed((os.getpid() * int(time.time())) % 123456789)


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> name
    print('> ' + os.path.basename(__file__))
    # print(f'> cwd = {cwd}')
    
    #> counting CPUs
    print(f'> There are {mp.cpu_count()} cores available!')

    #> command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=None, help='# of multiprocessing workers')
    parser.add_argument('--numMocks', type=int, default=1000, help='# of mocks to create')
    args = parser.parse_args()


    #> multiprocessing method (depends on OS)
    if os.name != "nt":  mp.set_start_method('fork', force=True) # linux
    else: mp.set_start_method('spawn', force=True)               # windows


    #> optionally override workers globally if provided
    if args.workers is not None:
        print('> Set workers.')
        workers = args.workers
    else:
        workers = 1 # default # of workers
        
    #> numMocks print
    print(f'> Creating {args.numMocks} mocks!')


    #> starting time
    srt = time.time()
    
    #> declarations
    createObs = False
    mockFile = dir_path+'mockSamples-dt23.txt'
    observables = ['t23', 'dt23']
    
    #> running experiment !
    arg = args.numMocks, workers, mockFile, observables, createObs
    exp3(arg)

    #> timing
    ttl = time.time() - srt
    print(f'> total time is {ttl/60:.3f} minutes')
    print(f'> ttl time/ mock = {ttl/args.numMocks:.3f} seconds')
    
    
    # end
# thank