# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:38:42 2021

@author: Filipe Fortuna
"""

import numpy as np


def SinalSequencia(nstep, a_range, b_range, type='RGS', random_state=14072021):
    
    '''
    Gera sinal de entrada.
    
    nstep: number of samples
    a_range: range for amplitude
    b_range: range for period
    
    '''
    
    # random signal generation
    np.random.seed(random_state)
    a = np.random.rand(nstep) * (a_range[1]-a_range[0]) + a_range[0]
    b = np.random.rand(nstep) *(b_range[1]-b_range[0]) + b_range[0]
    b = np.round(b).astype(int)
    
    b[0] = 0    
    for i in range(1,np.size(b)):
        b[i] = b[i-1]+b[i]
     
    
    if type=='PRBS':
        # PRBS
        a = np.zeros(nstep)
        j = 0
        while j < nstep:
            a[j] = 1
            a[j+1] = -1
            j = j+2
        
        i=0
        prbs = np.zeros(nstep)
        while b[i]<np.size(prbs):
            k = b[i]
            prbs[k:] = a[i]
            i=i+1
        sequence = prbs    

    
    if type=='RGS':
        # Random Signal
        i=0
        random_signal = np.zeros(nstep)
        while b[i]<np.size(random_signal):
            k = b[i]
            random_signal[k:] = a[i]
            i=i+1
            if i>=np.size(random_signal):
                break
        sequence = random_signal
    
    
    return sequence