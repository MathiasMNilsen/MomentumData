import os
import shutil
from glob import glob
import numpy as np

# import popt
from popt.loop.ensemble import Ensemble
from simulator.opm import flow
from input_output import read_config
from popt.update_schemes.enopt import EnOpt
from popt.cost_functions.npv import npv
from popt.misc_tools.optim_tools import time_correlation, corr2cov


def run_optimization(step_size, momentum, opt_type, save_folder):

    # Check if folder contains any En_ files, and remove them!
    for folder in glob('En_*'):
        shutil.rmtree(folder)
    for file in glob('optimize_result_*'):
        os.remove(file)

    if opt_type == 'Nesterov':
        optimizer = 'GA'
        nesterov  = True
    elif opt_type == 'Momentum':
        optimizer = 'GA'
        nesterov  = False
    else:
        optimizer = opt_type
        nesterov  = False

    save_folder = f'{save_folder}/{opt_type}_beta{momentum}'

    # settings
    settings = {'alpha'       : step_size,
                'beta'        : momentum,
                'nesterov'    : nesterov,
                'optimizer'   : optimizer,
                'save_folder' : save_folder}
    
    # Read config file
    ko, kf, ke = read_config.read_toml('init_optim.toml') 
    ko.update(settings)

    # Initalize ensemble
    ensemble = Ensemble(ke, flow(kf), npv)

    x0     = ensemble.get_state()
    cov    = ensemble.get_cov()
    bounds = ensemble.get_bounds()
    
    # add time correlation to cov
    corr = time_correlation(0.5,ensemble.state,10)
    cov  = corr2cov(corr, std=np.sqrt(np.diag(cov)))


    def diff_grad(x, *args):
        eps = 1e-3

        # Evaluate f(x) 
        fx = ensemble.function(x)

        # Evaluate f(x + eps)
        dim = x.size
        x_eps = np.zeros((dim, dim))
        for n in range(dim):
            
            # pertubation
            pert = np.zeros(dim)
            pert[n] = eps

            # add pertubation
            x_eps[:,n] = x + pert
        
        x_eps  = np.clip(x_eps, 0, 1)
        fx_eps = ensemble.function(x_eps)
        
        # approx gradient
        gradient = (fx_eps-fx)/eps
    
        return gradient

    def no_hessian(x=None, *args):
        return np.eye(len(x0))

    # Optimize
    EnOpt(ensemble.function, x0, args=(cov,), jac=diff_grad, hess=no_hessian, bounds=bounds, **ko)

    # save final injrate
    np.save('final_state.npy', ensemble.get_final_state())

    # Move stuff into folder
    files_to_move = ['opt_state.npz', 'final_state.npz', 'init_state.npz']

    for file in files_to_move:
        if os.path.isfile(file):
            shutil.move(file, save_folder)


if __name__ == '__main__':
    
    save_folder  = 'results'
    step_size    = 0.1
    methods      = ['Momentum', 'Nesterov', 'Adam', 'AdaMax']

    # initial state
    nSteps     = 10
    nInjWells  = 8
    nProdWells = 4
    injrate0   = 80
    prodrate0  = 100

    np.savez('init_injrate.npz' , injrate0*np.ones(nSteps*nInjWells))

    for beta in [0.0, 0.2, 0.5, 0.9]:
        for method in methods:

            run_optimization(step_size, beta, method, save_folder)
