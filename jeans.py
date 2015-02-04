from __future__ import division
from pylab import *
from scipy.integrate import quad
from scipy.special import gamma
from phot import *
import numpy as np
import time
import emcee
start_time = time.time()
np.random.seed(123) # Riproducible results

G = 4.302 * 10**-6 # Gravitational constant in Kiloparsec * solar_mass**-1 * (km/s)**2
dist=28.05         # Distance in Megaparsec
kpc_per_arcsec = dist*10**3 * pi / 3600. / 180. # Scale in kiloparsec per arcsec

''' data.dat is a text file which contains the data to be fitted. The format is:
1. Galactocentric radius in arcsec
2. Velocity dispersion in km/s
3. Velocity dispersion error in km/s'''
        
Rd,obs,err = loadtxt('data.dat',unpack=True)
N=len(Rd)
Rd = Rd * kpc_per_arcsec # converting arcsec to kpc

''' Here you input the photometric parameters of the tracer '''

R_e = 13.6                        # effective radius in kpc
n = 4.67                          # Sersic index
I0 = 2.84e9 /kpc_per_arcsec**2    # Central intensity in L_Sun / kpc^2
b = 2.*n - 1./3. + 0.009876/n     # b_n from Ciotti & Bertin (1999)
a_s = R_e / b**n                  # redefined scale radius
L = 2.*pi*n*gamma(2.*n)*I0*a_s**2 # total luminosity of the galaxy

# - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - 


# Redefining density profiles to more convenient form
def I_s(r): return I_Sersic(L,n,R_e,r) 
def j_s(r): return nu_Sersic(L,n,R_e,r)     
def L_s(r): return L_r_Sersic(L,n,R_e,r)
    
def sigm(Robs,rs,rhos,beta,A,gamma):
    '''
    This function returns the solution to equation A15 from Mamon & Lokas 2005
    
    The scale radius "rs" and the scale density "rhos" must be logarithmic quantities.
    "beta" is the constant anisotropy in the form :  - log(1-beta)
    "A" is stellar mass-to-light ratio in some band, 
    "gamma" is the inner-slope of the dark matter density profile   
    '''   
    
    rs=10**rs             # transforming rs to linear scale
    rhos=10**rhos         # transforming rhos to linear scale
    G = 4.302 * 10**-6
    beta=1. - 10**(-beta) # transforming beta to linear scale
    
    integrand = lambda r :  K_b(r/Robs , beta) * j_s(r) * (Md(r,rs,rhos,gamma) + A*L_s(r)) /r
    return sqrt( 2. * G * quad(integrand,Robs,inf)[0] / I_s(Robs) )

def lnlike(par,Rd,obs,err):  
    '''
    This function returns the log-likelihood function for a Gaussian distribution
    '''    
    rs,rhos,beta,A,gamma = par
    i=0
    model = zeros(N)
    for x in Rd : 
        model[i]= sigm(x,rs,rhos,beta,A,gamma)
        i+=1
    return -0.5*np.sum(((obs-model)**2 / err**2) + np.log(2* pi *err**2))


def lnprior(par):
    '''
    Prior function. You should set the range inside which the likelihood function
    is non-zero.
    '''
    rs, rhos, beta, A, gamma = par
    if 0 < rs < 3 and 5 < rhos < 9 and -0.6 < beta < 0.99 and 4. < A < 11.2 and 0. < gamma < 2. :
        return 0.0
    return -np.inf
    
   
def lnprob(par, x, y, yerr):
    '''
    Final probability distribution
    '''
    lp = lnprior(par)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(par, Rd, obs, err)

'''
Initial guess for MCMC
'''   

rs_in=2.
rhos_in=6.
beta_in=0.5
A_in=7.
gamma_in=1.

'''
Setting up the random walks and the sampler
'''  

ndim  = 5     # Number of dimensions
nwalkers = 40 # Number of walkers
nsteps = 5000 # Number of steps

pos = [[rs_in, rhos_in, beta_in, A_in, gamma_in] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(Rd, obs, err), threads=1)

print("Running MCMC...")
pos, prob, state = sampler.run_mcmc(pos, nsteps) 

print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))
                
end_time=(time.time() - start_time) /3600.
print ' --- completed in  %5.3f hours ' % end_time
savetxt('output.dat',sampler.flatchain,fmt='%2f')

sys.exit()
