import os
import numpy as np
import pandas as pd
import argparse
import emcee
import afterglowpy as grb
from astropy.cosmology import Planck15 as cosmo
from multiprocessing import Pool
import time
import datetime
np.random.seed(12)

# define params for afterglowpy
def get_GRB_params(params, z):
    dl = cosmo.luminosity_distance(z).cgs.value
    GRB_params =  {'jetType': grb.jet.TopHat, 
            'specType': 0, 
            'thetaObs': params[0],
            'E0':  float(10**(params[1])),
            'thetaCore': params[2], 
            'n0':  float(10**(params[3])), 
            'p': params[4],
            'epsilon_e':  float(10**(params[5])), 
            'epsilon_B':  float(10**(params[6])), 
            'xi_N': 1,
            'd_L': dl,
            'z': z}
    return GRB_params


# define params for mcmc
def get_mcmc_params(params, z):
    dl = cosmo.luminosity_distance(z).cgs.value
    mcmc_params = {'jetType': grb.jet.TopHat,
                'specType':    0,
                'thetaObs':    params[0],
                'E0':          params[1],
                'thetaCore':   params[2],
                'n0':          params[3],
                'p':           params[4],
                'epsilon_e':   params[5],
                'epsilon_B':   params[6],
                'xi_N':        1, # Fractions of electrons accelerated
                'd_L':         abs(dl),
                'z':           z
                }
    return mcmc_params


Params_to_fit = {
    'Fit': np.array(['thetaCore', 'thetaObs', 'E0', 'n0', 'p', 'epsilon_e',
                     'epsilon_B'])}


Params_Bound = {
    'thetaObs': np.array([0.0001, 1.57]),
    'E0': np.array([45, 60]),
    'n0': np.array([-10, 0]),
    'epsilon_e': np.array([-4, 0]),
    'epsilon_B': np.array([-20, 0]),
    'thetaCore': np.array([0.0001, 1.57]),
    'p': np.array([2.001, 4]),
    'thetaWing': np.array([0.1, 1.57]), ##Extra Paramter for Jet Structure Gaussian
    'b': np.array([0, 10]), # Power Law Index for Energy with theta inside the jet
    'z': np.array([0, 8])
    }


# Define the prior probability distribution
def LogPrior(params, z, Params_to_fit, Params_Bound):
    mcmc_params = get_mcmc_params(params, z)
    logprior = 0
    for prm in mcmc_params:
        if prm in Params_to_fit['Fit']:
            if prm == 'thetaObs':
                if (Params_Bound["thetaObs"][0] < mcmc_params[prm] <= Params_Bound["thetaObs"][1]):
                    if (mcmc_params[prm] < mcmc_params["thetaCore"]):
                        logprior += np.log(np.sin(mcmc_params[prm]))
                    else:
                        logprior += -np.inf
                else:
                    logprior += -np.inf
            else:
                if Params_Bound[prm][0] < mcmc_params[prm] <= Params_Bound[prm][1]:
                    logprior += -np.log((Params_Bound[prm][1]) - (Params_Bound[prm][0]))
                else:
                    return -np.inf
        else:
            logprior += 0
    return logprior


# define likelihood and cal posterior
def Loglikelihood(params, z, obs_time, obs_nu, obs_flux, obs_flux_err):
    GRB_params = get_GRB_params(params, z)
    # get flux from afterglowpy
    model_flux = grb.fluxDensity(obs_time, obs_nu, **GRB_params)
    # Calculate the log-likelihood
    residual = obs_flux - model_flux
    chi2 = np.sum(residual**2 / obs_flux_err**2)
    ln_like = -0.5 * chi2
    return ln_like


def logposterior(params, z, Params_to_fit, Params_Bound, obs_time, obs_nu, obs_flux, obs_flux_err):
    mcmc_params = get_mcmc_params(params, z)
    GRB_params = get_GRB_params(params, z)
    lp = LogPrior(mcmc_params, Params_Bound, Params_to_fit)
    if np.isfinite(lp):
        return lp + Loglikelihood(GRB_params, obs_time, obs_nu, obs_flux, obs_flux_err, z)
    else:
        return -np.inf

    
def best_fit_param(sampler):
    tau = sampler.get_autocorr_time(tol=0) #Auto-correlation time is the time after which emcee coverges
    burnin = int(2 * np.max(tau)) # data upto this time will be ignored
    thin = int(0.5 * np.min(tau))
    flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
    median_params = np.median(flat_samples, axis=0)
    return median_params

def full_run(params, Nwalker, n_burn, N_steps, logposterior_args):
    p0 = [params + 1.e-3 * np.random.randn(Ndim) for _ in range(Nwalker)]
    Ndim = len(params)
    sampler_filename = f"fullrun_initial_{datetime.datetime.now().strftime('%y%m%d_%H-%M-%S')}.h5"
    backend = emcee.backends.HDFBackend(sampler_filename)
    backend.reset(Nwalker, Ndim)
    
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalker, Ndim, logposterior, args=logposterior_args,  backend=backend, pool=pool)
        start = time.time()
        pos_, _, _ = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
        sampler.reset()
        sampler.run_mcmc(pos_, N_steps, progress=True)
        end = time.time()
        multi_time = end - start
        print(f"Multiprocessing took for first run is {multi_time:.1f} seconds")
    
    median_params = best_fit_param(sampler)
    # rerun with best fit param
    p0 = [median_params + 1.e-4 * np.random.randn(Ndim) for _ in range(Nwalker)]
    Ndim = len(params)
    sampler_filename = f"fullrun_median_{datetime.datetime.now().strftime('%y%m%d_%H-%M-%S')}.h5"
    backend = emcee.backends.HDFBackend(sampler_filename)
    backend.reset(Nwalker, Ndim)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalker, Ndim, logposterior, args=logposterior_args,  backend=backend, pool=pool)
        start = time.time()
        pos, _, _ = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
        sampler.reset()
        sampler.run_mcmc(pos, N_steps, progress=True)
        end = time.time()
        multi_time = end - start
        print(f"Multiprocessing took for first run is {multi_time:.1f} seconds")
    return

def single_run(params, Nwalker, n_burn, N_steps, logposterior_args):
    p0 = [params + 1.e-6 * np.random.randn(Ndim) for _ in range(Nwalker)]
    sampler_filename = f"single_run_{datetime.datetime.now().strftime('%y%m%d_%H-%M-%S')}.h5"
    backend = emcee.backends.HDFBackend(sampler_filename)
    backend.reset(Nwalker, Ndim)
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(Nwalker, Ndim, logposterior, args=logposterior_args,  backend=backend, pool=pool)
        start = time.time()
        pos_, _, _ = sampler.run_mcmc(p0, n_burn, store=False, progress=True)
        sampler.reset()
        sampler.run_mcmc(pos_, N_steps, progress=True)
        end = time.time()
        multi_time = end - start
        print(f"Multiprocessing took for first run is {multi_time:.1f} seconds")
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-obs_csv", help="csv of obs data with path", default=None)
    parser.add_argument(
        "-z", help="redshift", default=None)
    parser.add_argument(
        "-full_run", help="for first time run", default='N')
    parser.add_argument(
        "-single_run", help="if you have best fit params of full run", default='N')
    parser.add_argument(
        "-Nwalker", help="no of walker to search around each at steps", default=100)
    parser.add_argument(
        "-n_burn", help="steps to burn before mcmc run", default=10000)
    parser.add_argument(
        "-N_steps", help="no of steps for mcmc", default=50000)
    
    parser.add_argument(
        "--thetaObs", help="Viewing angle in radians", default=0.0069)
    parser.add_argument(
        "--E0", help="put exponent of 10, Isotropic-equivalent energy (erg)", default=56.35)
    parser.add_argument(
        "--thetaCore", help="Half-opening angle in radians", default=0.012)
    parser.add_argument(
        "--n0", help="put exponent of 10, circumburst density cm^{-3}", default=-1.537)
    parser.add_argument(
        "--p", help="electron energy distribution index", default=2.92)
    parser.add_argument(
        "--epsilon_e", help="put exponent of 10, epsilon_e", default=-0.817)
    parser.add_argument(
        "--epsilon_B", help="put exponent of 10, epsilon_B", default=-6.822)
    
    args = parser.parse_args()
    file = args.obs_csv
    path = os.path.basename(file) + '/'
    z = args.z
    Nwalker = args.Nwalker
    n_burn = args.n_burn
    N_steps = args.N_steps
    params = [args.thetaObs, args.E0, args.thetaCore, args.n0, args.p, args.epsilon_e, args.epsilon_B]
    Ndim = len(params)
    
    # read obs csv
    data = pd.read_csv(file)
    obs_time = data['Times']
    obs_nu = data['Freqs'].apply(lambda x: int(round(float(x))))
    obs_flux = data['Fluxes']
    obs_flux_err = data['FluxErrs']
    
    # dict of logposteroir args
    logposterior_args = {
        'z': z,
        'Params_to_fit': Params_to_fit,
        'Params_Bound': Params_Bound,
        'obs_time': obs_time,
        'obs_nu': obs_nu,
        'obs_flux': obs_flux,
        'obs_flux_err': obs_flux_err
        }
    
    print('--------------------------------')
    print('redshift : ', z)
    print('Nwalker : ', Nwalker)
    print('n_burn : ', n_burn)
    print('N_steps : ', N_steps)
    print('Ndim : ', Ndim)
    print('--------------------------------')
    
    if args.full_run == 'N' and args.single_run == 'N' :
        print('please choose full run or single run from parser')
    
    if args.full_run == 'Y':
        print('started full run mcmc')
        print('initial params : ', params)
        full_run(params, Nwalker, n_burn, N_steps, logposterior_args)
        
    if args.single_run == 'Y':
        print('started single run mcmc')
        print('updated params : ', params)
        single_run(params, Nwalker, n_burn, N_steps, logposterior_args)
    
    print('--------------------------------')
