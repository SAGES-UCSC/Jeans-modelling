from __future__ import division
from pylab import *
from scipy.integrate import quad
import numpy as np
np.seterr(all='warn')
from scipy.special import gamma, betainc,gammainc, beta, hyp2f1, hyp1f1
from scipy.integrate import quad

def gammai(a,x) :   
    ''' Redefining the incomplete Gamma function 
    in terms of hypergeometric function'''
    return x**a * a**-1 * hyp1f1(a,1.+a,-x)
       
def betai(a,b,x) :  
    ''' Redefining the incomplete Beta function 
    in terms of hypergeometric function'''
    return x**a * a**-1 * hyp2f1(a,1.-b,a+1.,x)

    
def K_b(u,beta) :
    ''' Jeans kernel for constant beta.
    This is equation A16 from Mamon & Lokas 2005'''
    global K
    if beta==+0.5 or beta==-0.5 :  
       K = u**(2.*beta -1.) * arccosh(u) * beta * sqrt(1.-1./u**2.)
    else: 
       K = 0.5*u**(2.*beta -1.) * ((1.5-beta)*sqrt(pi)*gamma(beta-0.5)/gamma(beta) + \
       beta*betai(beta+0.5,0.5,u**-2.) - betai(beta-0.5,0.5,u**-2.))
    return K
        
def Md(r,a,s,g) :
    ''' Cumulative Mass profile from a generalized Hernquist density profile '''
    return ( (4. * pi * s * a**3) / (3. - g) ) * (r/a)**(3-g) * hyp2f1(3-g, 3-g, 4-g, -r/a)
    
def MdLOG(r,v0,r0) :
    ''' Cumulative Mass profile from a logarithmic (LOG) density profile '''
    return (v0**2 * r**3)/(r0**2 + r**2)
    
def I_Sersic(L,n,R_e,R) :
    ''' Sersic surface density at radius R, for total luminosity L,
        index n, and effective radius R_e. '''
    b = 2.*n - 1./3.0 + 0.009876/n
    a_S = R_e / b**n                                         # Sersic scale radius
    I_0 = L / (2.*pi * n * R_e**2 / b**(2.*n) * gamma(2.*n)) # central value
    I = I_0 * exp(-(R/a_S)**(1./n))
    return I


def I_Sersic_gc(Ne,n,R_e,R) :
    ''' Sersic surface density at radius R, for total luminosity L,
        index n, and effective radius R_e. '''
    b = 2.*n - 1./3. + 0.009876/n
    I = Ne * exp(- b * ((R/R_e)**(1/n) -1))
    return I 
    
def nu_Sersic_gc(Ne,n,R_e,r) :
    ''' Sersic volume density at radius r, for total luminosity L,
        index n, and effective radius R_e. '''
    b = 2.*n - 1./3. + 0.009876/n
    a_S = R_e / b**n                    # Sersic scale radius     
    x = r / a_S
    I_0 = Ne * exp(- b * ((0./R_e)**(1. /n) -1.))
    p = 1. - 0.6097/n + 0.05463/n**2
    l1 = gamma(2.*n)/gamma((3.-p)*n) * I_0 / (2. * a_S)
    nu = l1 * exp(-x**(1./n)) / x**p
    return nu
    
def nu_Sersic(L,n,R_e,r) :
    ''' Sersic volume density at radius r, for total luminosity L,
        index n, and effective radius R_e. '''

    b = 2.*n - 1./3. + 0.009876/n
    a_S = R_e / b**n                    # Sersic scale radius     
    x = r / a_S
    p = 1.0 - 0.6097/n + 0.05463/n**2
    nu_0 = L / (4.*pi*n*gamma((3.-p)*n)*a_S**3)
    nu = nu_0 * exp(-x**(1./n)) / x**p
    
    return nu

def L_r_Sersic(L,n,R_e,r) :
    ''' Sersic cumulative luminosity at 3D radius r, for total luminosity L,
        index n, and effective radius R_e '''    
    b = 2.*n - 1./3. + 0.009876/n
    a_S = R_e / b**n                
    x = r / a_S
    p = 1.0 - 0.6097/n + 0.05463/n**2
    L_r = L * gammai((3.-p)*n,x**(1./n)) / gamma((3.-p)*n)
    
    return L_r
