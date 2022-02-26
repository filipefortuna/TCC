# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 15:20:15 2021

@author: Filipe Fortuna
"""

import numpy as  np

    
def CSTR_modificado(t,x,u,params={}):
    
    '''modelo do reator CSTR modificado'''
    
    # parametros do modelo
    Ca0 = 1.0     # mol/L
    T0 = 350         # K
    Tc0 = 350        # K
    V = 100          # L
    ha = 7e5         # cal/min.K
    delta_H = -2e5   # cal/mol
    k0 = 7.2e10      # /min
    E = 82724.3
    R = 8.314
    rho = 1000      # g/L
    rho_c = 1000    # g/L
    cp = 1.0        # cal/g.K
    cp_c = 1.0      # cal/g.K

    # variáveis de estado
    Ca, T = x
    
    # variáveis de entrada
    q = u[0]
    qc = u[1]
    
    # equações do modelo
    
    Ra = k0*Ca*np.exp(-E/(R*T))
    dCa = (q/V)*(Ca0-Ca) - Ra
    dT = (q/V)*(T0-T) - (Ra*delta_H)/(rho*cp) + ((rho_c*cp_c)/(rho*cp*V))*qc*(1-np.exp(-ha/(qc*rho_c*cp_c)))*(Tc0-T)
    
    # saida do modelo
    dx = [dCa, dT]
    
    return dx

def medida(t, x, u, params={}):
    
    """ Função de medida   """    
    cA, T = x
          
    return x