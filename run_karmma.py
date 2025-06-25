import sys
import pickle
import numpy as np
import h5py as h5
import healpy as hp
from karmma import KarmmaSampler, KarmmaConfig
from karmma.utils import *
import karmma.transforms as trf
from scipy.stats import norm, poisson
import torch

torch.set_num_threads(16)

configfile = sys.argv[1]
config     = KarmmaConfig(configfile)

nside    = config.analysis['nside']
gen_lmax = 3 * nside - 1
lmax     = 2 * nside

N_Z_BINS = config.analysis['nbins']
shift    = config.analysis['shift']
vargauss = config.analysis['vargauss']
ng_average = config.analysis['ng_average']

cl     = config.analysis['cl'][:,:,:(gen_lmax + 1)]

pixwin = config.analysis['pixwin']

#============= Load data =======================
Ng_obs = config.data['Ng_obs']
mask   = config.data['mask']

assert nside==hp.npix2nside(mask.shape[0]), 'Problem with nside!'

print("Initializing sampler....")
sampler = KarmmaSampler(Ng_obs, mask, cl, ng_average,config.y_cl_training_data_path,config.vargauss_training_data_path,lmax, gen_lmax,pixwin=pixwin)
     
print("Done initializing sampler....")

samples, mcmc_kernel = sampler.sample(config.n_burn_in, config.n_samples,config.step_size, x_init=config.x_init,bg_init=config.bg_init,cosmo_init=config.cosmo_init)

def x2kappa(xlm_real,xlm_imag,theta):
    kappa_list = []
    xlm        = sampler.get_xlm(xlm_real, xlm_imag)
    y_cl       = sampler.cl_emu.predict(theta).reshape((1,N_Z_BINS,N_Z_BINS,-1))[0]
    vargauss   = sampler.vargauss_emu.predict(theta)[0]
    shift      = torch.ones(N_Z_BINS)
    mu         = torch.log(shift) - 0.5*vargauss
    ylm        = sampler.apply_cl(xlm, y_cl)
    
    for i in range(N_Z_BINS):
        k = torch.exp(mu[i] + trf.Alm2Map.apply(ylm[i], nside, gen_lmax)) - shift[i]
        k = k.detach().numpy()
        k_filtered = get_filtered_map(k, sampler.pixwin_ell_filter.numpy(), nside)
        kappa_list.append(k_filtered)
    return np.array(kappa_list)

print("Saving samples...")

for i, (xlm_real, xlm_imag, bg, cosmo) in enumerate(zip(samples['xlm_real'], samples['xlm_imag'], samples['bg'],samples['cosmo'])):
    kappa = x2kappa(xlm_real,xlm_imag,cosmo)
    with h5.File(config.io_dir + '/sample_%d.h5'%(i), 'w') as f:
        f['i']          = i
        f['delta_g']    = kappa
        f['xlm_real']   = xlm_real
        f['xlm_imag']   = xlm_imag
        f['bg']         = bg
        f['cosmo']      = cosmo

print("Saving MCMC meta-data and mass matrix...")

with h5.File(config.io_dir + '/mcmc_metadata.h5', 'w') as f:
    f['step_size'] = mcmc_kernel.step_size
    f['num_steps'] = mcmc_kernel.num_steps

with open(config.io_dir + "/mass_matrix_inv.pkl","wb") as f:
    pickle.dump(mcmc_kernel.inverse_mass_matrix, f)
