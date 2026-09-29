#> param_correl.py
# the precursor to deflections.py
    
""" #> IMPORTS =======================
================================== """

#> reg imports
import os
import sys
import glob
from pathlib import Path
import time
import numpy as np
import pandas as pd
import scipy as sp
from copy import deepcopy

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

#> modules
import generate
import params
import modules.error as error
import modules.parse as parse
import modules.geometry as geometry
from modules.units import u; u = u()

#> directory path
global dir_path
dir_path = './data/param_correl/'


""" #> ALL VARIABLE FNS ==============
================================== """

#> returns paramRanges, mu, & cov for nfw_axis_ratio
def nfw_axisrat(bins, fit=False):
    
    #> parameter name
    name = 'nfw_axisrat'

    #> getting paramRanges
    galProfiles = generate.galProfiles(mult=[], ex=False)
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'axisrat':  {'init': 0.9, 'min':0.65, 'max':0.95, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if key in ['nfw'] and param in ['axisrat']:
                    dic[param] = vals[param]
                    continue
                if key in ['hern'] and param in ['axisrat']:
                    dic[param]['init'] = 0.95  # hernquist set constant to this value
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> sets std based on % of parameter range
    std_range_perc = min(100/bins, 10)
    
    #> determining std and saving boundaries
    key = list(vals.keys())[0]
    lb = vals[key]['min']
    ub = vals[key]['max']
    rng = ub - lb
    std = (rng * std_range_perc / 100)
        
    #> mu, cov
    width = rng / bins
    mu = np.arange(lb + width/2, ub + width/2, width)
    mu = np.array([[x] for x in mu])
    cov = np.identity(1) * std
        
    #> if not fitting, then zero out covariance matrix
    if not fit: cov = cov * 1e-50
                    
    return name, paramRanges, mu, cov


#> returns paramRanges, mu, & cov for hern_axis_ratio
def hern_axisrat(bins, fit=False):
    
    #> parameter name
    name = 'hern_axisrat'

    #> getting paramRanges
    galProfiles = generate.galProfiles(mult=[], ex=False)
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'axisrat':  {'init': 0.9, 'min':0.65, 'max':0.95, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if key in ['hern'] and param in ['axisrat']:
                    dic[param] = vals[param]
                    continue
                if key in ['nfw'] and param in ['axisrat']:
                    dic[param]['init'] = 0.95  # nfw set constant to this value
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> sets std based on % of parameter range
    std_range_perc = min(100/bins, 10)
    
    #> determining std and saving boundaries
    key = list(vals.keys())[0]
    lb = vals[key]['min']
    ub = vals[key]['max']
    rng = ub - lb
    std = (rng * std_range_perc / 100)
        
    #> mu, cov
    width = rng / bins
    mu = np.arange(lb + width/2, ub + width/2, width)
    mu = np.array([[x] for x in mu])
    cov = np.identity(1) * std
        
    #> if not fitting, then zero out covariance matrix
    if not fit: cov = cov * 1e-50
                    
    return name, paramRanges, mu, cov


#> returns paramRanges, mu, & cov for nfw_axis_ratio, with hern_axis_ratio='p01'
def both_axisrat(bins, fit=False):
    
    #> parameter name
    name = 'both_axisrat'

    #> getting paramRanges
    galProfiles = generate.galProfiles(mult=[], ex=False)
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'axisrat':  {'init': 0.9, 'min':0.65, 'max':0.95, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if key in ['nfw'] and param in ['axisrat']:
                    dic[param] = vals[param]
                    continue
                if key in ['hern'] and param in ['axisrat']:
                    dic[param]['init'] = 'p01'  # hern set equal to nfw axis ratio
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> sets std based on % of parameter range
    std_range_perc = min(100/bins, 10)
    
    #> determining std and saving boundaries
    key = list(vals.keys())[0]
    lb = vals[key]['min']
    ub = vals[key]['max']
    rng = ub - lb
    std = (rng * std_range_perc / 100)
        
    #> mu, cov
    width = rng / bins
    mu = np.arange(lb + width/2, ub + width/2, width)
    mu = np.array([[x] for x in mu])
    cov = np.identity(1) * std
        
    #> if not fitting, then zero out covariance matrix
    if not fit: cov = cov * 1e-50
                    
    return name, paramRanges, mu, cov


#> returns paramRanges, mu, & cov for nfw_theta
def nfw_theta(bins, fit=False):
    
    #> parameter name
    name = 'nfw_theta'

    #> getting paramRanges
    galProfiles = generate.galProfiles(mult=[], ex=False)
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'theta':  {'init': 0.0, 'min': 0.0, 'max': 90.0, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if key in ['nfw'] and param in ['theta']:
                    dic[param] = vals[param]
                    continue
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> sets std based on % of parameter range
    std_range_perc = min(100/bins, 10)
    
    #> determining std and saving boundaries
    key = list(vals.keys())[0]
    lb = vals[key]['min']
    ub = vals[key]['max']
    rng = ub - lb
    std = (rng * std_range_perc / 100)
        
    #> mu, cov
    width = rng / bins
    mu = np.arange(lb + width/2, ub + width/2, width)
    mu = np.array([[x] for x in mu])
    cov = np.identity(1) * std
        
    #> if not fitting, then zero out covariance matrix
    if not fit: cov = cov * 1e-50
                    
    return name, paramRanges, mu, cov


#> returns paramRanges, mu, & cov for m1_norm
def m1_norm(bins, fit=False):
    
    #> parameter name
    name = 'm1_norm'

    #> getting paramRanges
    galProfiles = generate.galProfiles(mult=[1], ex=False)
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'norm':  {'init':1e-2, 'min': 0.0, 'max': 0.05, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if key in ['mult'] and param in ['norm']:
                    dic[param] = vals[param]
                    continue
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> sets std based on % of parameter range
    std_range_perc = min(100/bins, 10)
    
    #> determining std and saving boundaries
    key = list(vals.keys())[0]
    lb = vals[key]['min']
    ub = vals[key]['max']
    rng = ub - lb
    std = (rng * std_range_perc / 100)
        
    #> mu, cov
    width = rng / bins
    mu = np.arange(lb + width/2, ub + width/2, width)
    mu = np.array([[x] for x in mu])
    cov = np.identity(1) * std
        
    #> if not fitting, then zero out covariance matrix
    if not fit: cov = cov * 1e-50
                    
    return name, paramRanges, mu, cov


#> returns paramRanges, mu, & cov for m1_theta
def m1_theta(bins, fit=False):
    
    #> parameter name
    name = 'm1_theta'

    #> getting paramRanges
    galProfiles = generate.galProfiles(mult=[1], ex=False)
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'theta':  {'init': 0.0, 'min': 0.0, 'max': 90,  'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if key in ['mult'] and param in ['theta']:
                    dic[param] = vals[param]
                    continue
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> sets std based on % of parameter range
    std_range_perc = min(100/bins, 10)
    
    #> determining std and saving boundaries
    key = list(vals.keys())[0]
    lb = vals[key]['min']
    ub = vals[key]['max']
    rng = ub - lb
    std = (rng * std_range_perc / 100)
        
    #> mu, cov
    width = rng / bins
    mu = np.arange(lb + width/2, ub + width/2, width)
    mu = np.array([[x] for x in mu])
    cov = np.identity(1) * std
        
    #> if not fitting, then zero out covariance matrix
    if not fit: cov = cov * 1e-50
                    
    return name, paramRanges, mu, cov


#> returns paramRanges, mu, & cov for m3_norm
def m3_norm(bins, fit=False):
    
    #> parameter name
    name = 'm3_norm'

    #> getting paramRanges
    galProfiles = generate.galProfiles(mult=[3], ex=False)
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'norm':  {'init':1e-2, 'min': -0.01, 'max': 0.01, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if key in ['mult'] and param in ['norm']:
                    dic[param] = vals[param]
                    continue
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> sets std based on % of parameter range
    std_range_perc = min(100/bins, 10)
    
    #> determining std and saving boundaries
    key = list(vals.keys())[0]
    lb = vals[key]['min']
    ub = vals[key]['max']
    rng = ub - lb
    std = (rng * std_range_perc / 100)
        
    #> mu, cov
    width = rng / bins
    mu = np.arange(lb + width/2, ub + width/2, width)
    mu = np.array([[x] for x in mu])
    cov = np.identity(1) * std
        
    #> if not fitting, then zero out covariance matrix
    if not fit: cov = cov * 1e-50
                    
    return name, paramRanges, mu, cov


#> returns paramRanges, mu, & cov for m1_norm
def m4_norm(bins, fit=False):
    
    #> parameter name
    name = 'm4_norm'

    #> getting paramRanges
    galProfiles = generate.galProfiles(mult=[4], ex=False)
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'norm':  {'init':1e-2, 'min': -0.01, 'max': 0.01, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if key in ['mult'] and param in ['norm']:
                    dic[param] = vals[param]
                    continue
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> sets std based on % of parameter range
    std_range_perc = min(100/bins, 10)
    
    #> determining std and saving boundaries
    key = list(vals.keys())[0]
    lb = vals[key]['min']
    ub = vals[key]['max']
    rng = ub - lb
    std = (rng * std_range_perc / 100)
        
    #> mu, cov
    width = rng / bins
    mu = np.arange(lb + width/2, ub + width/2, width)
    mu = np.array([[x] for x in mu])
    cov = np.identity(1) * std
        
    #> if not fitting, then zero out covariance matrix
    if not fit: cov = cov * 1e-50
                    
    return name, paramRanges, mu, cov


#> returns paramRanges, mu, & cov for m1_norm
def exshear(bins, fit=False):
    
    #> parameter name
    name = 'exshear'

    #> getting paramRanges
    galProfiles = generate.galProfiles(mult=[], ex=True)
    paramRanges = params.toggleParams(galProfiles)
    
    #> setting the wanted params ranges (init does nothing)
    vals = {'norm':  {'init':5e-2, 'min': 0, 'max': 0.05, 'fit': True},
            'theta': {'init': 0.0,  'min':-180, 'max': 180, 'fit': True}}
    
    #> changing all fit=False, except m1 norm, theta
    for key in paramRanges.keys():
        if key in ['zl', 'zs', 'pix_arc']: continue
        for dic in paramRanges[key]:
            for param in dic.keys():
                if key in ['ex'] and param in ['norm', 'theta']:
                    dic[param] = vals[param]
                    continue
                if type(dic[param]) == type({}):
                    dic[param]['fit'] = False
                    
    #> sets std based on % of parameter range
    std_range_perc = min(100/bins, 10)
    
    # # # # # NEEDS TO BE SET UP TO HANDLE BOTH NORM AND THETA
    
    #> determining std and saving boundaries
    key = list(vals.keys())[0]
    lb = vals[key]['min']
    ub = vals[key]['max']
    rng = ub - lb
    std = (rng * std_range_perc / 100)
        
    #> mu, cov
    width = rng / bins
    mu = np.arange(lb + width/2, ub + width/2, width)
    mu = np.array([[x] for x in mu])
    cov = np.identity(1) * std
        
    #> if not fitting, then zero out covariance matrix
    if not fit: cov = cov * 1e-50
                    
    return name, paramRanges, mu, cov



""" #> MAIN CREATION FN ==============
================================== """

#> creates a population of lenses based on shit
def createPop(numGals, observables, fn, bins, fit):
    
    #> globals
    global dir_path

    #> other declarations
    zl = 0.5
    zs = 1.0
    factor = 1
    nph = 50 * factor
    pix_arc = 60 * factor
    numSource_gal = 1
    
    #> grid declarations
    xgrid, ygrid = generate.grid(nph=nph)
    
    #> generating galaxy populations
    redshifts = np.array([(zl, zs)] * numGals)
    pix_arc = [pix_arc] * numGals
    
    #> sampling from specific function (specific param, see above)
    name, paramRanges, muv, cov = fn(bins, fit)
    
    #> creating all observed populations
    for i, mu in enumerate(muv):
        
        #> name of observed sample
        print(f'> Generating {name}')
        
        #> batch profiles
        bprofiles = params.bprofiles(numGals, paramRanges, 
                                     zl=redshifts[:,0], zs=redshifts[:,1], pix_arc=pix_arc,
                                     verbose=True, cov=cov, mu=mu)
        
        #for b in bprofiles:
        #    print(b['nfw'][0]['axisrat'])
        
        #> calculating deflection angles
        lenses = generate.genGalPop(redshifts, galProfiles=[], bprofiles=bprofiles)
        
        #> lensing sources
        images, _ = generate.genQuadPop(lenses, zs, jims=5,
                                              pix_arc=pix_arc, verbose=False,
                                              numSource_gal=numSource_gal)
        
        #> saving quad population
        if fit:
            filePath = dir_path + name + '_fit/' 
            Path(filePath).mkdir(parents=True, exist_ok=True)
            fileName = filePath + f'{name}_fit_{i+1}'
        else:
            filePath = dir_path + name + '/' 
            Path(filePath).mkdir(parents=True, exist_ok=True)
            fileName = filePath + f'{name}_{i+1}'
        np.save(fileName + '_positions', images)
        
        #> getting lensing observables
        img_observables = []
        for im in images:
            img_observables.append(geometry.lensObs((0, 0), im, observables))
        img_observables = np.array(img_observables)
        np.save(fileName + '_observables', img_observables)
        
        #> saving galaxy params
        gal = {'positions': fileName+'_positions.npy', 'observables': fileName+'_observables.npy',
               'headers': observables,
               'muv': mu, 'cov': cov, 'paramRanges': paramRanges}
        np.save(fileName + '_gal', gal)


""" #> CORRELATIONS ==================
================================== """

#> finding correlations between parameters
def correlations(folder):
    
    import re
    from natsort import natsorted
    
    #> getting .npy files from requested folder
    files = glob.glob(folder+'/*_gal.npy')
    files = sorted(files, key=lambda x: int(re.findall("(?<=)\d+", x)[0]))
    files = natsorted(files)
    numBins = int( len(files) / 2 )
    name = folder.split('/')[-1]
    
    #> iterating throuhg each file
    moments, mus, covs = [], [], []
    for file in files:
        
        #> loading data
        gal_data = np.load(file, allow_pickle=True).item()
        
        #> separating info
        positions = np.load(gal_data['positions'], allow_pickle=True)
        # observables = np.load(gal_data['observables'], allow_pickle=True)
        observables = np.load(gal_data['observables'].replace('_observables', '_new_observables'), allow_pickle=True)[:,:-1]
        headers = gal_data['headers']
        muv = gal_data['muv']
        cov = gal_data['cov']
        
        #> removing outliers
        df_obs = pd.DataFrame(observables, columns=headers)
        observables = df_obs[ df_obs['t12'] < 200 ].to_numpy()
        
        numObs = len(gal_data['headers'])
        dummy = []
        for i in range(numObs):
            dummy.append(np.mean(observables[:,i]))
            dummy.append(np.std(observables[:,i]))
            dummy.append(sp.stats.skew(observables[:,i]))
            dummy.append(sp.stats.kurtosis(observables[:,i]))
        
        #> saving
        moments.append(np.array(dummy))
        mus.append(muv)
        covs.append(cov)
    moments = np.array(moments)
    # print(moments)
    mus = np.array(mus)
    numMoments = int(len(dummy)/len(headers))
    
    mus = [x[0] for x in mus]
    
    #> getting pearson correlation coeffs
    moment_names = ['mu', 'sigma', 'skew', 'kurtosis']
    r_vals = []
    for i in range(numMoments*len(headers)):
        
        #> getting names
        head = headers[ int(i/len(moment_names)) ]
        moment_name = moment_names[ int( i % len(moment_names) ) ]
        
        #> getting percent change
        c_gal = ( mus[-1] - mus[0] ) / mus[0] * 100
        c_obs = ( moments[:,i][-1] - moments[:,i][0] ) / moments[:,i][0] * 100
        print(moments[:,i][0], moments[:,i][-1])
        
        #> r value and p values
        rval, pval = sp.stats.pearsonr(mus, moments[:,i])
        r_vals.append([name, head, moment_name, c_gal, c_obs, rval, pval])
    
    #> constructing dataframe
    df = pd.DataFrame(r_vals, columns=['gal_param', 'obs_param', 'moment', '%c_gal', '%c_obs', 'r_val', 'p_val'])
    df['flag'] = ( df['r_val'].apply(abs) > 0.5 ) & ( df['p_val'] < 0.05 )
    print(df)
    
    #> saving
    df.to_csv(folder+f'/{name}_correlations.csv', index=False)
    
    #> plotting
    plotCorrel(numObs, numMoments, moments, mus, headers, name)

        
    return


#> plotting the correlations
def plotCorrel(numObs, numMoments, moments, mus, headers, name):
    
    import matplotlib.pyplot as plt
    
    #> initializing plot
    fig, axes = plt.subplots(1,2, figsize=(6*2+1,6))
    
    #> plotting
    fontsize=15
    lw = 2
    ls = ['-', '--', '-.', ':']
    c=['r', 'm', 'b', 'c', 'g', 'tab:orange', 'k']
    alpha = 0.8
    for i in range(numObs):
        
        num = i*numMoments
        for j in range(numMoments):
            
            data = moments[:,num+j]
            ax = axes.flat[int(j/2)]
            if j == 0: 
                p1 = ax.plot(mus, (data - np.mean(data)) / np.mean(data), 
                             ls=ls[j], lw=lw, alpha=alpha, label=headers[i], c=c[i])
            else:
                p2 = ax.plot(mus, (data - np.mean(data)) / np.mean(data), 
                             ls=ls[j], lw=lw, alpha=alpha, c=p1[0].get_color())
            
            ax.grid(ls=':', alpha=0.5)
            ax.set_xlabel(f'{name}', fontweight='bold', labelpad=10, fontsize=fontsize)
            if j == 0 : ax.set_ylabel(r'$\mathbf{(M_i - E[M]) ~/ ~E[M]}$', fontsize=fontsize)
    
    #> adding extra legend
    axes.flat[1].plot([],[],ls=ls[0], c='k', label=r'$\mu$')
    axes.flat[1].plot([],[],ls=ls[1], c='k', label=r'$\sigma$')
    axes.flat[1].plot([],[],ls=ls[2], c='k', label=r'$S$')
    axes.flat[1].plot([],[],ls=ls[3], c='k', label=r'$K$')
    
    #> legend, saving, showing, closing
    axes.flat[0].legend()
    axes.flat[1].legend()
    plt.savefig(f'./figures/{name}_correl.png', bbox_inches='tight', dpi=100)
    plt.show(); plt.close()
    
    return


#> overplotting all observables
def plotObservables(folder):
    
    import re
    import matplotlib.pyplot as plt
    from natsort import natsorted
    
    #> getting .npy files from requested folder
    files = glob.glob(folder+'/*_gal.npy')
    files = sorted(files, key=lambda x: int(re.findall("(?<=)\d+", x)[0]))
    files = natsorted(files)
    numBins = int( len(files) / 2 )
    name = folder.split('/')[-1]
    
    #> initializing plot
    fig, axes = plt.subplots(2,3,figsize=(3*6, 2*6))
    alpha = 0.6
    lw=2
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
    
    plt.suptitle(f'{name}', y=0.92, fontweight='bold', fontsize=20)
    
    #> iterating throuhg each file
    moments, mus, covs = [], [], []
    for j, file in enumerate(files):
        
        #> loading data
        print(file)
        gal_data = np.load(file, allow_pickle=True).item()
        
        #> separating info
        positions = np.load(gal_data['positions'], allow_pickle=True)
        # observables = np.load(gal_data['observables'], allow_pickle=True)
        observables = np.load(gal_data['observables'].replace('_observables', '_new_observables'), allow_pickle=True)[:,:-1]
        headers = gal_data['headers']
        muv = gal_data['muv']
        cov = gal_data['cov']
        
        #> removing outliers
        df_obs = pd.DataFrame(observables, columns=headers)
        observables = df_obs[ df_obs['t12'] < 200 ].to_numpy()
        
        for i, ax in zip(range(len(headers)), axes.flat):
            n, bins, patches = ax.hist(observables[:,i], histtype='step', 
                                       alpha=alpha, lw=lw, density=True, label=f'{muv[0]:.3f}')  
            ax.set_yticklabels([])
            ax.axvline(np.mean(observables[:,i]), ls='--', c=patches[0].get_edgecolor())
            if j == 0:
                ax.grid(ls=':', alpha=0.5)
                ax.set_xlabel(headers[i], fontweight='bold', fontsize=15, labelpad=15)
    axes.flat[3].legend()
    
    print(f'./figures/{name}_histograms.png')
    plt.savefig(f'./figures/{name}_histograms.png', bbox_inches='tight', dpi=100)
    plt.show(); plt.close()
    
    return


#> plotting 
def fsq2D(folder):
    
    import re
    import matplotlib.pyplot as plt
    from natsort import natsorted
    
    #> getting .npy files from requested folder
    files = glob.glob(folder+'/*_gal.npy')
    files = sorted(files, key=lambda x: int(re.findall("(?<=)\d+", x)[0]))
    files = natsorted(files)
    numBins = int( len(files) / 2 )
    name = folder.split('/')[-1]
    
    #> initializing plot
    fig, axes = plt.subplots(1,2,figsize=(2*6, 6))
    alpha = 0.6
    lw=2
    
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
    
    plt.suptitle(f'{name}', y=0.92, fontweight='bold', fontsize=20)
    
    #> iterating throuhg each file
    moments, mus, covs = [], [], []
    for j, file in enumerate(files):
        
        #> loading data
        print(file)
        gal_data = np.load(file, allow_pickle=True).item()
        
        #> separating info
        # observables = np.load(gal_data['observables'], allow_pickle=True)
        observables = np.load(gal_data['observables'].replace('_observables', '_new_observables'), allow_pickle=True)[:,:-1]
        headers = gal_data['headers']
        muv = gal_data['muv']
        cov = gal_data['cov']
        
        #> removing outliers
        df_obs = pd.DataFrame(observables, columns=headers)
        observables = df_obs[ df_obs['t12'] < 200 ].to_numpy()
        
        for i, ax in zip(range(len(headers)), axes.flat):
            
            #> plotting dt23 hist
            if i == 0: 
                n, bins, patches = ax.hist(observables[:,-1], histtype='step', alpha=alpha, lw=lw, density=True, label=f'{muv[0]:.3f}')  
                ax.set_yticklabels([])
            
            if i == 1:
                ax.scatter(observables[:,1],observables[:,-1], alpha=0.05)
                
            if j == 0:
                if i == 0:
                    ax.set_xlabel(headers[-1], fontweight='bold', fontsize=15, labelpad=15)
                if i == 1:
                    ax.set_xlabel(headers[1], fontweight='bold', fontsize=15, labelpad=15)
                    ax.set_ylabel(headers[-1], fontweight='bold', fontsize=15, labelpad=15)
                ax.grid(ls=':', alpha=0.5)
                
    axes.flat[0].legend()
    
    print(f'./figures/{name}_histograms.png')
    plt.savefig(f'./figures/{name}_histograms.png', bbox_inches='tight', dpi=100)
    plt.show(); plt.close()
    
    return


""" #> RECALCULATES ==================
================================== """

#> recalculates observables
def recalObs(folder):
    
    #> imports
    import re
    sys.path.insert(1, 'C:/Users/mille/Desktop/Research/+projects/opus/modules/')
    import geometry
    import lensing
    from natsort import natsorted
    
    #> getting positions.npy files from requested folder
    files = glob.glob(folder+'/*_positions.npy')
    files = sorted(files, key=lambda x: int(re.findall("(?<=)\d+", x)[0]))
    files = np.array([x.replace('\\', '/') for x in files])
    files = natsorted(files)
    
    #> recalculates the observables for set of image positions
    observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23', 'd01']
    
    #> iterating thru files
    for file in files:
        
        #> loading images
        images = np.load(file, allow_pickle=True)
        
        #> new outfile name
        outFile = file[:-len('positions.npy')] + 'new_observables.npy'
        print(outFile)
        
        #> getting observables
        im_obs = []
        for ims in images:
            
            ims = np.array(ims, dtype=float)
            
            #> redoing order
            ordered_ims = lensing.arrivalOrder(ims, x0=0., y0=0.)
            
            #> getting observables
            im_obs.append(geometry.lensObs((0, 0), ordered_ims, observables))
            
        #> saving
        np.save(outFile, im_obs)
        
    return

#> compares differences between obsrevable foldesr
def compareObs(folder):
    
    #> imports
    sys.path.insert(1, '../opus_lmfi/')
    import re
    import geometry
    import lensing
    from natsort import natsorted
    
    #> getting positions.npy files from requested folder
    files = glob.glob(folder+'/*_new_observables.npy')
    files = sorted(files, key=lambda x: int(re.findall("(?<=)\d+", x)[0]))
    files = np.array([x.replace('\\', '/') for x in files])
    files = natsorted(files)
    
    #> getting the old files
    old_files = []
    for file in files:
        old_files.append(file.replace('_new', ''))
    old_files = np.array(old_files)
    
    #> loading and comparing the observables
    for new_file, file in zip(files, old_files):
        
        #> loading data
        new_obs = np.load(new_file, allow_pickle=True)[:,:-1]
        old_obs = np.load(file, allow_pickle=False)
        
        #> not equal
        mask = abs(new_obs - old_obs) > 0.001
        num_diff = sum(mask)
        print(num_diff)
    
    return


""" #> LOOKING AT OUTLIERS ===========
================================== """

#> looking at dt23 outliers
def dt23Outliers(folder):
    
    #> imports 
    import re
    import plot
    import matplotlib.pyplot as plt
    from natsort import natsorted
    from scipy import stats
    
    #> getting positions.npy files from requested folder
    files = glob.glob(folder+'/*_new_observables.npy')
    files = sorted(files, key=lambda x: int(re.findall("(?<=)\d+", x)[0]))
    files = np.array([x.replace('\\', '/') for x in files])
    files = natsorted(files)
    
    observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1' ,'d4/d1', 'dt23', 'd01']
    
    #> iterating thru files
    for file in files:
        
        #> loading obs
        im_obs = np.load(file, allow_pickle=True)
        df = pd.DataFrame(im_obs, columns=observables)
        
        #> image positions
        positions = np.load(file.replace('_new_observables', '_positions'), allow_pickle=True)
        
        #> getting z score
        df['z'] = np.abs(stats.zscore(df['dt23']))
        
        #> getting outliers
        outliers = df[ df['z'] > 10.0 ]['dt23']
        
        #> if there are outliers
        if len(outliers) != 0:
            
            #> print info
            print(file)
            print(outliers)
            
            #> get positions
            indx = outliers.index.to_numpy()
            outlier_pos = positions[indx]
            
            #> plot quads
            for ims in outlier_pos:
                fig, ax = plt.subplots(1,1,figsize=(6,6))
                ax.grid(ls=':', alpha=0.5)
                for i, im in enumerate(ims):
                    ax.scatter(im[0], im[1], label=f'{i+1}')
                ax.scatter(0,0,marker='+', s=100, c='k')
                plot.toBox(ax)
                plot.ticks(ax)
                plt.legend()
                plt.show(); plt.close()
    
    return


""" #> W-COMPARISONS =================
================================== """

#> compares the param_correl gals and quads
def w_comp(folder):
    
    #> imports
    import re
    from natsort import natsorted
    import modules.metric as metric
    
    #> getting new_obseravables.npy files from requested folder
    files = glob.glob(folder+'/*_new_observables.npy')
    files = sorted(files, key=lambda x: int(re.findall("(?<=)\d+", x)[0]))
    files = np.array([x.replace('\\', '/') for x in files])
    obs_files = natsorted(files)
    
    #> getting gal files
    gal_files = np.array([x.replace('_new_observables', '_gal') for x in obs_files])
    
    #> define a 'observed' file
    
    for obs_file, gal_file in zip(obs_files, gal_files):
        
        gal = np.load(gal_file, allow_pickle=True).item()
        # print(gal.keys())
        # print(gal['paramRanges']['nfw'][0]['axisrat'])
    
    
    return


""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':
    
    #> name
    print('> '+os.path.basename(__file__))
    
    #> declarations
    commandLine = False
    
    #> command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type=str, default='nfw_axisrat', help='param name')
    parser.add_argument('--bins', type=int, default=10, help='# number of bins')
    parser.add_argument('--numGals', type=int, default=10000, help='# of gals to generate per bin')
    parser.add_argument('--fit', action='store_true', help='return plots')
    args = parser.parse_args()
    
    #> declarations
    bins = args.bins
    numGals = args.numGals
    fit = args.fit # whether to have spread in params or not
     #print(fit)
    functions = {'nfw_axisrat': nfw_axisrat, 'hern_axisrat': hern_axisrat, 
                 'both_axisrat': both_axisrat, 'nfw_theta': nfw_theta, 
                 'm1_norm': m1_norm, 'm1_theta': m1_theta, 
                 'm3_norm': m3_norm, 'm4_norm': m4_norm}
    observables = ['t12', 't23', 't34', 'd2/d1', 'd3/d1', 'd4/d1', 'dt23']
    
    #> for command line
    if commandLine:
        fn = functions[args.param]
        # createPop(numGals, observables, fn, bins, fit)
        #if fit:
        #    folder = dir_path + fn.__name__ + '_fit'
        #else:
        #    folder = dir_path + fn.__name__
        #correlations(folder)
        #plotObservables(folder)
    else:
        for key in functions.keys():
            fn = functions[key]
            # if key != 'nfw_axisrat': continue
            # createPop(numGals, observables, fn, bins, fit)
            if fit:
                folder = dir_path + fn.__name__ + '_fit'
            else:
                folder = dir_path + fn.__name__
            #correlations(folder)
            #plotObservables(folder)
            w_comp(folder)
            # recalObs(folder)
            # compareObs(folder)
            # fsq2D(folder)
            # dt23Outliers(folder)
    
    # folder = dir_path + 'nfw_axisrat'
    # correlations(folder)
    # plotObservables(folder)

    # end
# thank