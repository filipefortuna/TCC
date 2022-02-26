# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 11:28:00 2021

@author: Filipe Fortuna
"""

import numpy as np

class ESN():
    def __init__(self, n_inputs, n_outputs,
                 Rsize=200 , Sparsity=0.0, Spectral_radius=0.95,
                 alpha=0.70, Ridge=0.0004, Noise_level=0.001,
                 random_state=14072021):
        
        """  
        inputs: matriz N x nu (variáveis de entrada)
        outputs: matriz N x ny (variáveis de saída)
        par: dicionário de parâmetros da rede Echo State
  
        """
        # parametros globais da rede
        self.n_inputs = n_inputs               # Entrada
        self.n_outputs = n_outputs
        self.Rsize = Rsize                     # Tamanho do reservatório (M)
        self.Sparsity = Sparsity               # Adicionar zeros nas matrizes de pesos
        self.Spectral_radius = Spectral_radius # Raio espectral
        self.alpha = alpha                     # Taxa de vazamento
        self.Ridge = Ridge                     # Ridge Regression (Bheta)
        self.Noise_level = Noise_level         # adicionar ruido
        self.random_state = random_state
        
        self.initpesos()


    def initpesos(self):
        # Inicialização dos pesos
        np.random.seed(self.random_state)
        #  começar com uma matriz centralizada no entorno de zero:
        W = np.random.rand(self.Rsize, self.Rsize) - 0.5
        # deletar a fraçao de conexões dada pela sparsity:
        W[np.random.rand(*W.shape) < self.Sparsity] = 0
        
        # computar o raio espectral desses pesos:
        Rho =  np.max(np.abs(np.linalg.eigvals(W)))
        # rescalar o pesos para o raio espectral exigido:
        self.W = W*(self.Spectral_radius/Rho)
        
        # pesos de entrada
        self.Win = np.random.rand(self.n_inputs+1, self.Rsize)*2 - 1
        
        # pesos do feedback
        self.Wfb = np.random.rand(self.n_outputs, self.Rsize)*2 - 1
        

    def normalization_inputs(self, inputs):
        # entrada
        self.minU = np.min(inputs, axis=0)
        self.maxU = np.max(inputs, axis=0)
        inputs = 0.8*(inputs-self.minU)/(self.maxU-self.minU) + 0.1
        
        return inputs
    
    def normalization_outputs(self, outputs):
        # saida
        self.minY = np.min(outputs, axis=0)
        self.maxY = np.max(outputs, axis=0)
        outputs = 0.8*(outputs-self.minY)/(self.maxY-self.minY) + 0.1
        
        return outputs
    
    def desnormalization(self, outputs):
        
        outputs = (outputs - 0.1)*(self.maxY - self.minY)/0.8 + self.minY
        
        return outputs
    

    def _update(self, state, part_inputs, part_outputs):
        
        preactivation = (state @ self.W 
                         + part_inputs @ self.Win 
                         + part_outputs @ self.Wfb)
        
        return (np.tanh(preactivation))

    
    def train(self, inputs, outputs):
        
        inputs = self.normalization_inputs(inputs)
        outputs = self.normalization_outputs(outputs)
        
        # adicionando bias à entrada da rede
        inputs = np.hstack([np.ones((len(inputs),1))*0.1, inputs])
        outputsfb = np.vstack([np.zeros((1,self.n_outputs)), outputs[0:-1,:]])
        
        state = np.zeros((inputs.shape[0], self.Rsize))
    
        for n in range(1, inputs.shape[0]):
            state[n,:] =  self._update(state[n-1], inputs[n,:], outputsfb[n-1,:])
            state[n,:] = (1-self.alpha)*state[n-1,:] + self.alpha*state[n,:]
      
        # treinamento da rede
        state = state + np.random.randn(*state.shape)*self.Noise_level
        extended_state = np.hstack([state, inputs])
    
        self.laststate = state[-1, :]
        self.lastinputs = inputs[-1, :]
        self.lastoutputs = outputsfb[-1, :]
    
        self.Wout = np.linalg.inv(extended_state.transpose() @ extended_state + self.Ridge*np.eye(extended_state.shape[1])) @  extended_state.transpose() @ outputs
        ytrain = extended_state @ self.Wout
        
        ytrain = self.desnormalization(ytrain)
        
        return ytrain
    
    def predict(self, inputs, continuation=True):
        
        if continuation:
            laststate = self.laststate
            lastinputs = self.lastinputs
            lastoutputs = self.lastoutputs
        else:
            laststate = np.zeros(self.Rsize)
            lastinputs = np.zeros(self.n_inputs+1)
            lastoutputs = np.zeros(self.n_outputs)
        
        inputs = self.normalization_inputs(inputs)
        n_samples = inputs.shape[0]
        inputs = np.hstack([np.ones((len(inputs),1))*0.1, inputs])
        
        inputs = np.vstack([lastinputs, inputs])
        state = np.vstack(
            [laststate, np.zeros((n_samples, self.Rsize))])
        outputs = np.vstack(
            [lastoutputs, np.zeros((n_samples, self.n_outputs))])
    
        for n in range(n_samples):
            
            state[n+1,:] =  self._update(state[n], inputs[n+1,:], outputs[n,:])
            state[n+1,:] = (1-self.alpha)*state[n,:] + self.alpha*state[n+1,:]
            
            outputs[n+1,:] = np.hstack([state[n+1,:], inputs[n+1,:]]) @ self.Wout
            
        outputs = self.desnormalization(outputs)
        outputs = outputs[1:,:]
        
        return outputs


        
    