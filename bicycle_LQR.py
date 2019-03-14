#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:58:23 2019

@author: shansenb
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
L1=3 #meters
L2=1.5 #meters

#%%
v = lambda t: t
delta = lambda t: np.sin(np.pi*t)

def bike_kinematic(Z,t):
    
    xdot= (v(t)/L1)*((L1*np.cos(Z[2]))-L2*np.sin(Z[2])*np.tan(delta(t)))
    ydot = (v(t)/L1)*((L1*np.sin(Z[2]))+L2*np.cos(Z[2])*np.tan(delta(t)))
    psidot = (v(t)/L1)*np.tan(delta(t))
    zdot = [xdot, ydot, psidot]
    
    return zdot

#simulate for 5 seconds at .01 second intervals
T=np.linspace(0,5,501)

#initial conditions
Z0=[ 0., 0., 0.]

Z=integrate.odeint(bike_kinematic ,Z0, T)

plt.plot(Z[:,0],Z[:,1])

#%%

gamma = lambda t: np.sin(np.pi*t) #redefine as gamma
psi = lambda t: (v(t)/L1)*np.tan(gamma(t))*t;

def Afun(t):
    
    A=np.matrix([[0., 0., -v(t)*np.sin(psi(t))-(v(t)*L2/L1)*np.cos(psi(t))*np.tan(gamma(t))], \
                 [0., 0.,  v(t)*np.cos(psi(t))-(v(t)*L2/L1)*np.sin(psi(t))*np.tan(gamma(t))], \
                 [0., 0.,                       0.                    ]])
    return A
    
    
def Bfun(t):
    
    B = np.matrix([[1/L1*(L1*np.cos(psi(t))-L2*np.sin(psi(t))*np.tan(gamma(t))), -np.cos(np.pi*t)*(v(t)*L2*np.sin(psi(t)))/(L1*np.cos(gamma(t))**2)], \
                    [1/L1*(L1*np.sin(psi(t))+L2*np.cos(psi(t))*np.tan(gamma(t))), np.cos(np.pi*t)*(v(t)*L2*np.cos(psi(t)))/(L1*np.cos(gamma(t))**2)],\
                    [1/L1*np.tan(gamma(t)), v(t)/(L1*np.cos(gamma(t))**2)]])
    
    return B

Q = np.diag([1.0, 1.0, 1.0])



R =np.diag([1.0, 1.0])


def LTV_LQR(AFun,BFun,Q,R,tSpan):
    nSteps = len(tSpan)

    #initialize matrices to hold covariance and gain
    P = np.zeros((Q.shape[0],Q.shape[1],nSteps))
    K= np.zeros((len(R),len(Q), nSteps))
    t = lambda i: tSpan[i]
    
    for i in range(nSteps-2,0,-1):
        #start at nSteps-1 go to step 0
        A_ = AFun(t(i+1));
        B_ = BFun(t(i+1));
        P_ = P[:,:,i+1];
        
        temp1 = np.linalg.lstsq(R,(B_.transpose()*P_))
        P[:,:,i] = P_ + (tSpan[i+1]-tSpan[i]) * ( P_*A_ + A_.transpose()*P_ - P_*B_*temp1[0] + Q)
        temp2 = np.linalg.lstsq(R,(B_.transpose()*P_))
        K[:,:,i] = temp2[0]
        
    return [K, P]

#%%

[K,P] = LTV_LQR(Afun,Bfun,Q,R,T)



#%%

#initial configuration q
x0=np.array(([0.1, 0.8, 0.1]))

Z= np.zeros((len(x0),len(T)))

for i in range(len(T)):
    Z = (Afun(T[i])-Bfun(T[i]))*K(:,:,i)*x0.transpose()
    
    













