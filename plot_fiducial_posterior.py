import sys
import numpy as np
import h5py as h5
import healpy as hp
from karmma import KarmmaSampler, KarmmaConfig
from karmma.utils import *
import torch
import pyro.poutine as poutine
# ====================================================================
torch.set_num_threads(16)

configfile = sys.argv[1]
config     = KarmmaConfig(configfile)

nside      = config.analysis['nside']
gen_lmax   = 3 * nside - 1
lmax       = 2 * nside
N_Z_BINS   = config.analysis['nbins']
ng_average = config.analysis['ng_average']
pixwin     = config.analysis['pixwin']
thetafid   = config.analysis['thetafid']
bfid       = config.analysis['bfid']
Ng_obs     = config.data['Ng_obs']
mask       = config.data['mask']
iodir      = config.io_dir
#============= Load data =======================
assert nside==hp.npix2nside(mask.shape[0]), 'Problem with nside!'

print("Initializing sampler....")
sampler = KarmmaSampler(Ng_obs, mask, ng_average,config.y_cl_training_data_path,config.vargauss_training_data_path,thetafid,bfid,lmax, gen_lmax,pixwin=pixwin)

def compute_log_prob(b,cosmo,xlm_real, xlm_imag):
    conditioned_model = poutine.condition(sampler.model,
        data={"bg": b,
              "cosmo": cosmo,
              "xlm_real": xlm_real, 
              "xlm_imag": xlm_imag}
        )
    trace = poutine.trace(conditioned_model).get_trace()
    log_prob = trace.log_prob_sum()  
    return log_prob
# ====== Load fiducial/truth data ==================================
with h5.File(config.datafile, 'r') as f:
    xlm_real_true = torch.tensor(f['xlm_real'][()],dtype=torch.double)
    xlm_imag_true = torch.tensor(f['xlm_imag'][()],dtype=torch.double)
    cosmo_true    = f['cosmo'][()]

    Om_true       = cosmo_true[0]
    S8_true       = cosmo_true[1]
    cosmo_true    = torch.tensor(f['cosmo'][()],dtype=torch.double)
    bg_true       = torch.tensor(f['bg'][()],dtype=torch.double)

# ====== Log-posterior functions ===================================
def logP_Om(Omega_m):
    # Ensure Omega_m requires gradients
    if not Omega_m.requires_grad:
        Omega_m = Omega_m.clone().detach().requires_grad_(True)
    
    # Create cosmo without breaking the computational graph
    cosmo = torch.zeros(2, dtype=torch.double)
    cosmo[0] = Omega_m  # This preserves the gradient connection
    cosmo[1] = S8_true
    
    # Calculate and return the log probability
    return compute_log_prob(bg_true, cosmo, xlm_real_true, xlm_imag_true)

def logP_S8(S8):
    # Ensure Omega_m requires gradients
    if not S8.requires_grad:
        S8 = S8.clone().detach().requires_grad_(True)
    
    # Create cosmo without breaking the computational graph
    cosmo = torch.zeros(2, dtype=torch.double)
    cosmo[0] = Om_true  # This preserves the gradient connection
    cosmo[1] = S8
    
    # Calculate and return the log probability
    return compute_log_prob(bg_true, cosmo, xlm_real_true, xlm_imag_true)
# ====== Plots ===============================================================
print('Creating plots')
N = 100  
Om_bounds = [0.98*Om_true,1.02*Om_true]
S8_bounds = [0.98*S8_true,1.02*S8_true]
plt.figure(figsize=(15, 5)) 
# =============================================================================
# Plot 1: Omega_m posterior
plt.subplot(1, 2, 1)
log_P_array = np.zeros(N)
Omega_m_array = np.zeros(N)
delta = (Om_bounds[1] - Om_bounds[0]) / N

for i in range(N):
    current_Om = Om_bounds[0] + i * delta
    Omega_m_array[i] = current_Om
    log_P_array[i] = logP_Om(torch.tensor(current_Om)).detach().numpy()

plt.plot(Omega_m_array, np.exp(log_P_array - log_P_array.max()))
plt.axvline(Om_true, color='black', linestyle='dashed', label='fiducial')
plt.xlabel(r'$\Omega_m$')
plt.ylabel(r'$\exp(\log P-\max(\log P))$')
plt.title(r'$\Omega_m$ Posterior')
plt.legend()
# =============================================================================
# Plot 2: S8 posterior
plt.subplot(1, 2 ,2)
log_P_array = np.zeros(N)
S8_array = np.zeros(N)
delta = (S8_bounds[1] - S8_bounds[0]) / N

for i in range(N):
    current_S8 = S8_bounds[0] + i * delta
    S8_array[i] = current_S8
    log_P_array[i] = logP_S8(torch.tensor(current_S8)).detach().numpy()

plt.plot(S8_array, np.exp(log_P_array - log_P_array.max()))
plt.axvline(S8_true, color='black', linestyle='dashed', label='fiducial')
plt.xlabel(r'$S_8$')
plt.ylabel(r'$\exp(\log P-\max(\log P))$')
plt.title(r'$S_8$ Posterior')
plt.legend()

plt.tight_layout()
plt.savefig(config.io_dir+'/posterior.png')
plt.show()
plt.close()