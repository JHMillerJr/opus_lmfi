#> name: opus.py
#> author: John Miller Jr
#> descrp: main python file for the lensing code Opus

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import time
import numpy as np
import scipy.stats as stats
from copy import deepcopy
import ot

import matplotlib.pyplot as plt

import multiprocessing as mp

#> checking to see if GPU is available
# if error.checkGPU(): import cupy as cp

#> declarations
dataDir = './data/'
figDir = './figures/'
sys.path.append(os.path.dirname(__file__)) 

dpi = 200


""" #> FUNCTIONS =====================
================================== """

#> generates a population of quads from a population of galaxies
def genQuadPops(M, M_G, observables=None, suffix='', profiles=None):
    
    #> imports
    import params
    import lensing
    import modules.error as error
    import modules.parse as parse
    import deflection # putting here due to cupy import issues (w/ parallel too in lensing.pfims())
    
    #> checking int
    if M % M_G != 0:
        error.phrase('M/M_G is not an int!')
    
    #> grid declarations
    nph = 50
    pix_arc = 60 # default in bprofiles
    w_range = np.arange(-nph, nph, 1, dtype=float)
    xgrid, ygrid = np.meshgrid(w_range, w_range)
    
    #> declarations
    dims = [0, 3, 6, 9]
    observables = ['t12', 't23', 't34', 'd4/d1', 'd3/d1', 'd2/d1', 'd1', 'dt23',
                   'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']
    times = {} # how long it takes to calculate everything
    
    #> iterating through each # of galaxy param dims
    for d in dims:
        
        paramRanges = params.paramRanges()
        
        #> adjusting params
        if d in [3, 6, 9]:
            
            paramRanges['nfw'][0]['axisrat']['fit'] = True
            paramRanges['nfw'][0]['theta']['fit'] = True
            paramRanges['hern'][0]['axisrat']['fit'] = True
            
            if d in [6, 9]:
                
                paramRanges['mult'][0]['norm']['fit'] = True
                paramRanges['mult'][0]['theta']['fit'] = True
                paramRanges['mult'][1]['norm']['fit'] = True
                
                if d in [9]:
                    
                    paramRanges['mult'][2]['norm']['fit'] = True
                    paramRanges['ex'][0]['norm']['fit'] = True
                    paramRanges['ex'][0]['theta']['fit'] = True
        
        #> all params have been changed
        profiles = params.bprofiles(M_G, paramRanges)
    
        #> calculating deflections
        srt = time.time()
        delx, dely, lamt = deflection.deflectionGPU(xgrid, ygrid, profiles)
        
        #> if cupy is available
        if error.checkGPU(): 
            import cupy as cp
            cp.cuda.Stream.null.synchronize() # if there is GPU available, can use
            delx = cp.asnumpy(delx) # converting arrays (for parallelization later)
            dely = cp.asnumpy(dely)
            lamt = cp.asnumpy(lamt)
              
        tg = time.time()-srt
        print(f'> Generateing galaxies took {time.time()-srt:.1e}s or {(time.time()-srt)/M_G:.1e}s/gal.')
        
        #> lensing sources
        pix_arc=np.full(shape=(1,M_G),fill_value=pix_arc)[0]
        srt = time.time()
        images = lensing.pfims(xgrid, ygrid, # pixels
                               delx, dely,   # radians
                               lamt,         # tangential critical curve
                               pix_arc, nph, # pix_arc, nph=width/2
                               int(M/M_G),   # number of quads per galaxy
                               observables)  # requesting lensing observables (if=None, returns image positions)
        tq = time.time() - srt
        print(f'> Generating sources took {time.time()-srt:.1e}s or {(time.time()-srt)/M:.1e}s/src.')
        
        #> saving times
        times[d] = {'t': tg+tq, 'tg': tg,'tq': tq}
        
        #> saving quads
        outFile = dataDir+time.strftime('%y%m%d%H%M', time.localtime()) + f'-{d}'
        
        np.save(outFile, images)        # quad pop
        parse.toJSON(outFile, profiles) # profiles
    parse.toJSON(dataDir+'dtimes', times)

    return images, tg, tq


#> randomly draws obs and mocks (ensures no overlap)
def ranObsAndMocks(quadPop, N, M):
    
    #> getting indicies
    indxes = np.arange(len(quadPop))
    obsIndx = np.random.choice(indxes, size=N, replace=False)
    
    indxes = [item for item in indxes if item not in obsIndx]
    mockIndx = np.random.choice(indxes, size=M, replace=False)
    
    obs = quadPop[obsIndx]
    mock = quadPop[mockIndx]
        
    return obs, mock


#> doing self comparisons between large populations
def comparison(file):
    
    #> loading in 'fake' data
    data = np.load(file)
    name = file.split('/')[-1].split('.')[0]
    
    #> declarations
    N_values = [1e2, 1e3]
    M_values = [1e2, 1e3, 1e4]
    # observables = ['t23', 'dt23', 'd4/d1']
    observables = ['t12', 't23', 't34', 'd4/d1', 'd3/d1', 'd2/d1', 'd1']
    ranges = np.array([180, 90, 180, 1, 1, 1, 1])
    
    #> SW declarations
    lb = 0
    ub = 4
    bins = 2
    L = np.arange(lb, ub+(ub/bins), ub/bins)
    print(L)
    
    #> iterating thru all population comparisons
    results = []
    for i, N in enumerate(N_values):          # obs population size
    
        obs, _ = ranObsAndMocks(data, int(N), 0)
    
        for j, M in enumerate(M_values):      # mock population size
        
            #> choosing obs and mock populations
            _, mock = ranObsAndMocks(data, 0, int(M))
            
            print(N, M)
            
            for k in range(len(observables)): # iterating thru all lensing observables
                for l in L:                   # getting all slices
                                        
                    #> calculating SW distance
                    times = []
                    sw_dists = []
                    for _ in range(10):
                        srt = time.time()
                        dist = ot.sliced_wasserstein_distance(obs[:,:k+1]/ranges[:k+1], mock[:,:k+1]/ranges[:k+1], n_projections=int(10**l))
                        sw_dists.append(dist)
                        times.append(time.time() - srt)
                    
                    #> calculating error on SW
                    lb, ub = stats.t.interval(confidence=0.95,           # confidence interval
                                              df=len(sw_dists)-1,        # degrees of freedom (sample size - 1)
                                              loc=np.mean(sw_dists),     # sample mean
                                              scale=stats.sem(sw_dists)) # stanrd error of the mean
                    mean_time = np.mean(times)
                    
                    row = [N, M, k+1, l, mean_time, lb, np.mean(sw_dists), ub]
                    results.append(row)
                    # print(N, M, k+1, l, mean_time, lb, np.mean(sw_dists), ub)
    
    np.save('./metrics/w-data/comp-norm-' + name, results)
    
    return results


""" #> NON-SELF-COMPS ================
================================== """

#> function
def allComp(args):
    
    obs, mock, observables, ranges, N, M, L = args
    
    results = []
    k = len(observables)
    
    for l in L:                   # getting all slices
                            
        #> calculating SW distance
        times = []
        sw_dists = []
        for _ in range(10):
            srt = time.time()
            dist = ot.sliced_wasserstein_distance(obs[:,:len(observables)]/ranges, mock[:,:len(observables)]/ranges, n_projections=int(10**l))
            sw_dists.append(dist)
            times.append(time.time() - srt)
        
        #> calculating error on SW
        lb, ub = stats.t.interval(confidence=0.95,           # confidence interval
                                  df=len(sw_dists)-1,        # degrees of freedom (sample size - 1)
                                  loc=np.mean(sw_dists),     # sample mean
                                  scale=stats.sem(sw_dists)) # stanrd error of the mean
        mean_time = np.mean(times)
        
        row = [N, M, k+1, l, mean_time, lb, np.mean(sw_dists), ub]
        print(row)
        results.append(row)
        # print(N, M, k+1, l, mean_time, lb, np.mean(sw_dists), ub)
            
    return np.array(results)


#> doing self comparisons between large populations
def allComparisons(files):
    
    #> declarations
    N_values = [1e2, 1e3]
    M_values = [1e2, 1e3, 1e4]
    observables = ['t12', 't23', 't34', 'd4/d1', 'd3/d1', 'd2/d1']
    ranges = np.array([180, 90, 180, 1, 1, 1])
    
    #> SW declarations
    lb = 0
    ub = 4
    bins = 2
    L = np.arange(lb, ub+(ub/bins), ub/bins)
    
    #> multiprocessing declarations
    workers = max(1, mp.cpu_count() - 1)
    
    #> opening pool
    
    for z, file1 in enumerate(files):
        
        data1 = np.load(file1)
        name1 = file1.split('/')[-1].split('.')[0]
        
        for x, file2 in enumerate(files):
            
            pool = mp.Pool(processes=workers)
            
            if z < x: continue # skipping upper triangle
            
            #> loading different data sets
            
            data2 = np.load(file2)
            name2 = file2.split('/')[-1].split('.')[0]
            
            print(z, name1)
            print(x, name2)
            
            # submit all tasks first
            async_results = []
            for i, N in enumerate(N_values):          # obs population size
            
                obs, _ = ranObsAndMocks(data1, int(N), 0)
                print(obs.shape)
            
                for j, M in enumerate(M_values):      # mock population size
                
                    #> choosing obs and mock populations
                    _, mock = ranObsAndMocks(data2, 0, int(M))
                    
                    #> running parallel
                    args = (obs, mock, observables, ranges, N, M, L)
                    async_results.append(pool.apply_async(allComp, args=(args,)))
                                         
            pool.close()
            
            #> collecting results
            results = []
            for r in async_results: results.append(r.get())
            pool.join()
            
            results = np.array(results)
            np.save('./metrics/w-data/comp-' + name1 + name2, results)
            
    return results


""" #> PLOTTING ======================
================================== """

#> plotting comparisons
def plotComps(inFile):
    
    import matplotlib.ticker as ticker
    
    #> loading data
    data = np.load(inFile)
    time, numParams = inFile.split('/')[-1].split('-')[2:]
    numParams = numParams[0]
    
    #> header info
    # N, M, # obs, l, mean_time, lb, np.mean(sw_dists), ub
    # 0  1  2      3  4          5   6                  7
    
    #> all varied galaxy parameters
    params = [r'$\mathbf{q_{\bullet}}$', r'$\mathbf{\theta_{\bullet}}$', r'$\mathbf{q_*}$',
              r'$\mathbf{a_1}$', r'$\mathbf{\theta_1}$', r'$\mathbf{a_3}$',
              r'$\mathbf{a_4}$', r'$\mathbf{\gamma}$', r'$\mathbf{\theta_\gamma}$']
    
    #> declarations
    N_values = [1e2, 1e3]
    M_values = [1e2, 1e3, 1e4, 1e5]
    observables = ['t12', 't23', 't34', 'd4/d1', 'd3/d1', 'd2/d1', 'd1']
    
    #> initializing plot
    fig, axes = plt.subplots(2,4,figsize=(6*4+2,13))
    fontsize=25
    
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.92, wspace=0.1, hspace=0.0)
    
    colors = ['green', 'blue']
    ls = ['-', '--', '-.', ':']
    
    allParams = ''
    for i in range(int(numParams)):
        allParams += params[i] + ', '
        
    plt.suptitle(f't={time}, ' + r'$\mathbf{\theta}$' + f'=[{allParams[:-2]}]', 
                 fontsize=30, fontweight='bold')
    
    #> plotting!
    for i, ax in enumerate(axes.flat):
        
        ax.grid(ls=':', alpha=0.8)
        
        for j, N in enumerate(N_values):
            for k, M in enumerate(M_values):
                
                #> plotting
                d = data[ (data[:,0]==N) & (data[:,1]==M) & (data[:,2]==i+1) ]
                ax.plot(d[:,3], d[:,6], color=colors[j], ls=ls[k], lw=3)
                ax.fill_between(d[:,3], d[:,5], d[:,7], color=colors[j], alpha=0.05)
        
        #> labels
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.set_ylim(0, 0.5)
        
        if i < 3:
            # ax.set_xticks([])
            ax.set_xticklabels([])
        if i > 2: 
            ax.set_xlabel(r'$\mathbf{log_{10}(L)}$', fontsize=fontsize, labelpad=10)
            
        if i in [0, 4]: 
            ax.set_ylabel(r'$\mathbf{SW}$', fontsize=fontsize, labelpad=10)
        
        if i<7: ax.annotate(f'+{observables[i]}', xy=(0.99, 0.94), 
                            xycoords='axes fraction', fontsize=fontsize, 
                            fontweight='bold',
                            horizontalalignment='right')
        
        #> last figure
        if i == 7:
            ax.axis('off')
            
        ax.plot([],[],color='green',label='N=1e2',lw=10)
        ax.plot([],[],color='blue',label='N=1e3',lw=10)
        ax.plot([],[],label=' ',color='w')
        ax.plot([],[],color='k',label='M=1e2',lw=4, ls='-')
        ax.plot([],[],color='k',label='M=1e3',lw=4, ls='--')
        ax.plot([],[],color='k',label='M=1e4',lw=4, ls='-.')
        ax.plot([],[],color='k',label='M=1e5',lw=4, ls=':')
            
        plt.legend(fontsize=30, bbox_to_anchor=(0.81, 0.86))
    
    plt.savefig(f'./metrics/w-figs/SW_selfcomp_{numParams}.png', dpi=dpi)
    plt.show()        
    
    return


#> error on SW versus time
def SWvtime(inFile):
    
    import matplotlib.ticker as ticker
    
    #> loading data
    data = np.load(inFile)
    time, numParams = inFile.split('/')[-1].split('-')[2:]
    numParams = numParams[0]
    
    #> header info
    # N, M, # obs, l, mean_time, lb, np.mean(sw_dists), ub
    # 0  1  2      3  4          5   6                  7
    
    #> all varied galaxy parameters
    params = [r'$\mathbf{q_{\bullet}}$', r'$\mathbf{\theta_{\bullet}}$', r'$\mathbf{q_*}$',
              r'$\mathbf{a_1}$', r'$\mathbf{\theta_1}$', r'$\mathbf{a_3}$',
              r'$\mathbf{a_4}$', r'$\mathbf{\gamma}$', r'$\mathbf{\theta_\gamma}$']
    
    #> declarations
    N_values = [1e2, 1e3]
    M_values = [1e2, 1e3, 1e4, 1e5]
    observables = ['t12', 't23', 't34', 'd4/d1', 'd3/d1', 'd2/d1', 'd1']
    
    #> initializing plot
    fig, axes = plt.subplots(2,4,figsize=(6*4+2,13))
    fontsize=25
    
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.92, wspace=0.1, hspace=0.0)
    
    colors = ['green', 'blue']
    ls = ['-', '--', '-.', ':']
    
    allParams = ''
    for i in range(int(numParams)):
        allParams += params[i] + ', '
        
    plt.suptitle(f't={time}, ' + r'$\mathbf{\theta}$' + f'=[{allParams[:-2]}]', 
                 fontsize=30, fontweight='bold')
    
    #> plotting!
    for i, ax in enumerate(axes.flat):
        
        ax.grid(ls=':', alpha=0.8)
        
        for j, N in enumerate(N_values):
            for k, M in enumerate(M_values):
                
                #> plotting
                d = data[ (data[:,0]==N) & (data[:,1]==M) & (data[:,2]==i+1) ]
                ax.plot(d[:,3], np.log10(d[:,4]), color=colors[j], ls=ls[k], lw=3)
                # ax.fill_between(d[:,3], d[:,5], d[:,7], color=colors[j], alpha=0.05)
        
        #> labels
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.set_ylim(-4, 3)
        
        if i < 3:
            # ax.set_xticks([])
            ax.set_xticklabels([])
        if i > 2: 
            ax.set_xlabel(r'$\mathbf{log_{10}(L)}$', fontsize=fontsize, labelpad=10)
            
        if i in [0, 4]: 
            ax.set_ylabel(r'$\mathbf{log_{10}(t)~~[s]}$', fontsize=fontsize, labelpad=10)
        
        if i<7: ax.annotate(f'+{observables[i]}', xy=(0.99, 0.94), 
                            xycoords='axes fraction', fontsize=fontsize, 
                            fontweight='bold',
                            horizontalalignment='right')
        
        #> last figure
        if i == 7:
            ax.axis('off')
            
        ax.plot([],[],color='green',label='N=1e2',lw=10)
        ax.plot([],[],color='blue',label='N=1e3',lw=10)
        ax.plot([],[],label=' ',color='w')
        ax.plot([],[],color='k',label='M=1e2',lw=4, ls='-')
        ax.plot([],[],color='k',label='M=1e3',lw=4, ls='--')
        ax.plot([],[],color='k',label='M=1e4',lw=4, ls='-.')
        ax.plot([],[],color='k',label='M=1e5',lw=4, ls=':')
            
        plt.legend(fontsize=30, bbox_to_anchor=(0.81, 0.86))
    
    plt.savefig(f'./metrics/w-figs/SW_selfcomp_time_{numParams}.png', dpi=dpi)
    plt.show()        
    
    return


#> sliced wasserstein error versus time
def SWEvtime(inFile):
    
    import matplotlib.ticker as ticker
    
    #> loading data
    data = np.load(inFile)
    time, numParams = inFile.split('/')[-1].split('-')[2:]
    numParams = numParams[0]
    
    #> header info
    # N, M, # obs, l, mean_time, lb, np.mean(sw_dists), ub
    # 0  1  2      3  4          5   6                  7
    
    #> all varied galaxy parameters
    params = [r'$\mathbf{q_{\bullet}}$', r'$\mathbf{\theta_{\bullet}}$', r'$\mathbf{q_*}$',
              r'$\mathbf{a_1}$', r'$\mathbf{\theta_1}$', r'$\mathbf{a_3}$',
              r'$\mathbf{a_4}$', r'$\mathbf{\gamma}$', r'$\mathbf{\theta_\gamma}$']
    
    #> declarations
    N_values = [1e2, 1e3]
    M_values = [1e2, 1e3, 1e4, 1e5]
    observables = ['t12', 't23', 't34', 'd4/d1', 'd3/d1', 'd2/d1', 'd1']
    
    #> initializing plot
    fig, axes = plt.subplots(2,4,figsize=(6*4+2,13))
    fontsize=25
    
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=0.92, wspace=0.1, hspace=0.0)
    
    colors = ['green', 'blue']
    ls = ['-', '--', '-.', ':']
    
    allParams = ''
    for i in range(int(numParams)):
        allParams += params[i] + ', '
        
    plt.suptitle(f't={time}, ' + r'$\mathbf{\theta}$' + f'=[{allParams[:-2]}]', 
                 fontsize=30, fontweight='bold')
    
    #> plotting!
    for i, ax in enumerate(axes.flat):
        
        ax.grid(ls=':', alpha=0.8)
        
        for j, N in enumerate(N_values):
            for k, M in enumerate(M_values):
                
                #> plotting
                d = data[ (data[:,0]==N) & (data[:,1]==M) & (data[:,2]==i+1) ]
                error = (d[:,7] - d[:,5]) / d[:,6]
                ax.plot(d[:,3], np.log10(error), color=colors[j], ls=ls[k], lw=3)
                # ax.fill_between(d[:,3], d[:,5], d[:,7], color=colors[j], alpha=0.05)
        
        #> labels
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.set_ylim(-2.5, 0)
        
        if i < 3:
            # ax.set_xticks([])
            ax.set_xticklabels([])
        if i > 2: 
            ax.set_xlabel(r'$\mathbf{log_{10}(L)}$', fontsize=fontsize, labelpad=10)
            
        if i in [0, 4]: 
            ax.set_ylabel(r'$\mathbf{log_{10}(95\% ~/ ~<SW>)}$', fontsize=fontsize, labelpad=10)
        
        if i<7: ax.annotate(f'+{observables[i]}', xy=(0.99, 0.94), 
                            xycoords='axes fraction', fontsize=fontsize, 
                            fontweight='bold',
                            horizontalalignment='right')
        
        #> last figure
        if i == 7:
            ax.axis('off')
            
        ax.plot([],[],color='green',label='N=1e2',lw=10)
        ax.plot([],[],color='blue',label='N=1e3',lw=10)
        ax.plot([],[],label=' ',color='w')
        ax.plot([],[],color='k',label='M=1e2',lw=4, ls='-')
        ax.plot([],[],color='k',label='M=1e3',lw=4, ls='--')
        ax.plot([],[],color='k',label='M=1e4',lw=4, ls='-.')
        ax.plot([],[],color='k',label='M=1e5',lw=4, ls=':')
            
        plt.legend(fontsize=30, bbox_to_anchor=(0.81, 0.86))
    
    plt.savefig(f'./metrics/w-figs/SW_selfcomp_error_{numParams}.png', dpi=dpi)
    plt.show()        
    
    return


#> plotting the lower triangle comparisons
def triComp():
    
    import matplotlib.ticker as ticker
    
    #> getting files
    dataDir = './metrics/w-data/comp-*.npy'
    files = glob.glob(dataDir)
    files = [ x.replace('\\', '/') for x in files ]
    
    #> collceting actual files
    dataFiles = []
    for file in files:
        if len(file) < 50: continue
        dataFiles.append(file)
    
    #> initializing plot
    fig, axes = plt.subplots(4,4,figsize=(24,24))
    
    #> all varied galaxy parameters
    params = [r'$\mathbf{q_{\bullet}}$', r'$\mathbf{\theta_{\bullet}}$', r'$\mathbf{q_*}$',
              r'$\mathbf{a_1}$', r'$\mathbf{\theta_1}$', r'$\mathbf{a_3}$',
              r'$\mathbf{a_4}$', r'$\mathbf{\gamma}$', r'$\mathbf{\theta_\gamma}$']
    
    #> declarations
    N_values = [1e2, 1e3]
    M_values = [1e2, 1e3, 1e4]
    observables = ['t12', 't23', 't34', 'd4/d1', 'd3/d1', 'd2/d1']
    colors = ['green', 'blue']
    ls = ['-', '--', '-.', ':']
    
    count = 0
    #> iterating through grid
    for i in range(4):
        for j in range(4):
            
            #> getting axis
            ax = axes[i][j]
            
            #> upper triangle grids
            if i < j: 
                ax.axis('off')
                continue
            
            #> data file
            file = dataFiles[count]
            name1, name2, numParams = file.split('/')[-1].split('-')[1:]
            
            data = np.load(file)
            data = data.reshape(18,8)
            print(name1, name2, numParams)
            print(count)
            count += 1
            
            ax.grid(ls=':', alpha=0.8)
            
            for p, N in enumerate(N_values):
                for k, M in enumerate(M_values):
                    
                    d = data[ (data[:,0]==N) & (data[:,1]==M) ]
                    print(d[:,3])
                    print(d[:,6])
                    ax.plot(d[:,3], d[:,6], color=colors[p], ls=ls[k], lw=3)
                    # ax.fill_between(d[:,3], d[:,5], d[:,7], color=colors[j], alpha=0.05)
        
            
            #> labels
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            # ax.set_ylim(-2.5, 0)
            
            # plt.show()
            # sys.exit()
            
    return



""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> generating the four quad populations
    M   = 1e4
    M_G = 1e4
    # genQuadPops(int(M), int(M_G))    
    
    import glob
    
    files = glob.glob('./metrics/w-data/2*.npy')
    files = [ x.replace('\\', '/') for x in files ]
    triComp()
    # allComparisons(files)
    
    sys.exit()
    
    # comparison()
    if len(sys.argv) > 1:
        args = sys.argv[1:]

        fileNum = int(args[0])

        file = files[fileNum]
        name = file.split('/')[-1]
        comparison(file)
        plotComps(f'./metrics/w-data/comp-{name}')
        sys.exit()
    
    sys.exit()
    for i, file in enumerate(files):
        # if i != 0: continue
        print(i, file)
        name = file.split('/')[-1]
        
        print(f'comp-norm-{name}')
        # comparison(file)
        plotComps(f'./metrics/w-data/comp-norm-{name}')
        SWvtime(f'./metrics/w-data/comp-norm-{name}')
        SWEvtime(f'./metrics/w-data/comp-norm-{name}')
    
    # end
# thank