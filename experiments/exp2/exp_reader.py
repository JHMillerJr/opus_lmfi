#> name: exp_reader.py

""" #> IMPORTS =======================
================================== """

#> standard imports
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import corner

#> main directory
global proj_loc
proj_loc = 'projects/'

#> modules
from modules.units import u; u=u()
import modules.error as error


""" #> OBS ===========================
================================== """

#> returns the observed params
def obs(col):
    
    muv = [[0.000, 0.0],
           [0.005, 0.0],
           [0.010, 0.0],
           [0.005, 45.],
           [0.010, 45.],
           [0.010, -45.],
           [0.010, 135.],
           [0.010, -135.],
           [0.010, 180]]
    return muv[col-1]


""" #> READING FILE ==================
================================== """

#> returns data from file
def read(fileName, names):
    return pd.read_csv(fileName, names=names, sep=' ')


""" #> GETTING POSTERIOR =============
================================== """

#> getting posterior
def post(file, names, numPost, col):
    
    #> getting data
    data = read(file, names)
    if 'dt23' in file: flag = '-dt23'
    else: flag = ''
    
    #> print
    print(f'> Creating posterior w/ {numPost/len(data)*100:.2f}% of samples')
    print(f'> Total number of samples is {len(data)}')
    
    #> parsing data
    labels = ['mu_a1', 'mu_t1', f'SW{col}']
    ndata = data[labels].copy()
    ndata = ndata.sort_values(by=[f'SW{col}']).to_numpy()[:numPost]
    
    #> corner
    corner.corner(ndata[:,:-1], labels=labels[:-1], truths=obs(col), truth_color='r',
                  show_titles=True, title_fmt='.3f', titles=[obs(col)[0], obs(col)[1]])
    
    outName = loc_dir+f'post-{col}'+flag+'.png'
    print(outName)
    plt.savefig(loc_dir+f'post-{col}'+flag+'.png', dpi=100, bbox_inches='tight')
    
    return
    



""" #> MAIN ==========================
================================== """

#> main function
if __name__ == '__main__':

    #> name
    print('> ' + os.path.basename(__file__))
    
    
    #> file and columns
    global loc_dir
    loc_dir = './metrics/exp2/'
    fileName1 = loc_dir + 'mockSamples.txt'
    fileName2 = loc_dir + 'mockSamples-dt23.txt'
    names = ['mu_a1', 'mu_t1', 'sigma_a1', 'sigma_t1', 'SW1', 'SW2', 'SW3', 'SW4', 'SW5', 'SW6', 'SW7', 'SW8', 'SW9']
    
    #> iterating through files
    for file in zip([fileName1, fileName2]):
        for i in range(9):
            post(file[0], names, 1000, i+1)
    
    
    # end
# thank