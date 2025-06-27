import karmma.mock as mock
from karmma import KarmmaSampler, KarmmaConfig
import sys
import numpy as np
import healpy as hp
import torch 
#====== Load
print('Loading from configfile...')
configfile     = sys.argv[1]
config         = KarmmaConfig(configfile)

nside          = config.analysis['nside']
nbins          = config.analysis['nbins']
npix           = hp.nside2npix(nside)
gen_lmax       = 3*nside - 1
lmax           = 2*nside 
cosmofid       = torch.tensor(config.thetafid,dtype=torch.double)
bfid           = config.bfid
N_gals_average = config.analysis['ng_average']
mask           = hp.fitsfunc.read_map(config.maskfile)
boolean_mask   = mask.astype(bool)

tmp            = np.zeros((nbins,npix))
#======= Initializing sampler
print('Initializing sampler...')
sampler = KarmmaSampler(tmp,tmp,N_gals_average,config.y_cl_training_data_path,config.vargauss_training_data_path,config.thetafid,config.bfid,lmax,gen_lmax,pixwin=config.analysis['pixwin'])
# ====== Creating mock data
print('Computing mocks...')
vargauss   = sampler.vargauss_emu.predict(cosmofid)[0].numpy()
shift      = np.ones(nbins)
mu         = np.log(shift) - 0.5*vargauss
ycl        = sampler.cl_emu.predict(cosmofid).reshape((1,nbins,nbins,-1))[0].numpy()
y_maps,xlm = mock.get_GRF(ycl,nside,nbins,gen_lmax)
delta_m    = mock.get_LNRF(y_maps,shift,mu)
delta_g    = mock.apply_bias(delta_m,bfid)
#delta_g_lp = mock.low_pass_filter(delta_g)
Ng         = mock.get_Ng(delta_g,N_gals_average,boolean_mask,sampler.pixwin_ell_filter.numpy())

# ====== Saving...
print('Saving...')
mock.save_datafile(Ng,mask,delta_g,config.thetafid,config.bfid,xlm,config.datafile)