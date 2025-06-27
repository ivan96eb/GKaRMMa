import numpy as np
import healpy as hp
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from .transforms import Alm2Map, Map2Alm
from .emulator import ClEmu,LNParamsEmu
import pickle
##==================================

class KarmmaSampler:
    def __init__(self, Ng_obs, mask, ng_average,y_cl_training_data_path,vargauss_training_data_path, thetafid,bfid,lmax=None, gen_lmax=None ,pixwin=None):
        self.ng_average = ng_average
        self.Ng_obs     = Ng_obs       
        self.N_Z_BINS   = Ng_obs.shape[0]
        self.mask       = mask.astype(bool)

        self.nside         = hp.get_nside(self.Ng_obs)
        self.lmax          = 2 * self.nside if not lmax else lmax
        self.gen_lmax      = 3 * self.nside - 1 if not gen_lmax else gen_lmax     
        self.ell, self.emm = hp.Alm.getlm(self.gen_lmax)

        self.thetafid = thetafid
        self.bfid     = bfid

        self.train_emulator(y_cl_training_data_path,vargauss_training_data_path)
        
        if pixwin is not None:
            print("Using healpix pixel window function.")
            from scipy.interpolate import interp1d

            ell_pixwin, _ = hp.Alm.getlm(self.lmax)
            if pixwin=='healpix':
                pixwin = hp.sphtfunc.pixwin(self.nside, lmax=self.gen_lmax)
            else:
                pixwin = pixwin
            pixwin_interp = interp1d(np.arange(len(pixwin)), pixwin)
            pixwin_ell_filter = pixwin_interp(ell_pixwin)
            self.pixwin_ell_filter = torch.tensor(pixwin_ell_filter)
        else:
            self.pixwin_ell_filter = None

        self.tensorize()
    
    def tensorize(self):
        self.Ng_obs   = torch.tensor(self.Ng_obs)
        self.mask     = torch.tensor(self.mask)
        self.thetafid = torch.tensor(self.thetafid,dtype=torch.double)
        self.bfid     = torch.tensor(self.bfid,dtype=torch.double)

    def train_emulator(self, ycl_file,vargauss_file):
        self.cl_emu = ClEmu(torch.load(ycl_file))
        self.cl_emu.fit_params()    

        self.vargauss_emu = LNParamsEmu(torch.load(vargauss_file))
        self.vargauss_emu.fit_params()

    def get_xlm(self, xlm_real, xlm_imag):
        ell, emm = hp.Alm.getlm(self.gen_lmax)
        _xlm_real = torch.zeros(self.N_Z_BINS, len(ell), dtype=torch.double)
        _xlm_imag = torch.zeros_like(_xlm_real)
        _xlm_real[:,ell > 1] = xlm_real
        _xlm_imag[:,(ell > 1) & (emm > 0)] = xlm_imag
        xlm = _xlm_real + 1j * _xlm_imag
        return xlm

    def matmul(self, A, x):
        y = torch.zeros_like(x)
        for i in range(self.N_Z_BINS):
            for j in range(self.N_Z_BINS):
                y[i] += A[i,j] * x[j]
        return y

    def apply_cl(self, xlm, cl):
        ell, emm = hp.Alm.getlm(self.gen_lmax)
        
        L = torch.linalg.cholesky(cl.T).T
    
        xlm_real = xlm.real
        xlm_imag = xlm.imag
        
        L_arr = torch.swapaxes(L[:,:,ell[ell > -1]], 0,1)
    

        ylm_real = self.matmul(L_arr, xlm_real) / torch.sqrt(torch.Tensor([2.]))
        ylm_imag = self.matmul(L_arr, xlm_imag) / torch.sqrt(torch.Tensor([2.]))

        ylm_real[:,ell[emm==0]] *= torch.sqrt(torch.Tensor([2.]))
    
        return ylm_real + 1j * ylm_imag
    
    def model(self, prior_only=False):
        ell, emm = hp.Alm.getlm(self.gen_lmax)

        xlm_real = pyro.sample('xlm_real', dist.Normal(torch.zeros(self.N_Z_BINS, (ell > 1).sum(), dtype=torch.double),
                                                       torch.ones(self.N_Z_BINS, (ell > 1).sum(), dtype=torch.double)))
        xlm_imag = pyro.sample('xlm_imag', dist.Normal(torch.zeros(self.N_Z_BINS, ((ell > 1) & (emm > 0)).sum(), dtype=torch.double),
                                                       torch.ones(self.N_Z_BINS, ((ell > 1) & (emm > 0)).sum(), dtype=torch.double)))
          
        xlm      = self.get_xlm(xlm_real, xlm_imag)

        cosmo    = pyro.sample('cosmo', dist.Uniform(torch.tensor([0.25, 0.7],dtype=torch.double),
                                                      torch.tensor([0.4, 0.9],dtype=torch.double)))
        
        bg       = pyro.sample('bg', dist.Uniform(0.7*torch.ones(self.N_Z_BINS, dtype=torch.double), 
                                            1.3*torch.ones(self.N_Z_BINS, dtype=torch.double)))
        
        vargauss = self.vargauss_emu.predict(cosmo)[0]
        shift    = torch.ones(self.N_Z_BINS)
        mu       = torch.zeros(self.N_Z_BINS)
        for i in range(self.N_Z_BINS):
            mu[i] = torch.log(shift[i]) - 0.5 * vargauss[i]

        y_cl = self.cl_emu.predict(cosmo).reshape((1,self.N_Z_BINS,self.N_Z_BINS,-1))[0]
        ylm  = self.apply_cl(xlm, y_cl)

        for i in range(self.N_Z_BINS):
            delta_m    = torch.exp(mu[i] + Alm2Map.apply(ylm[i], self.nside, self.gen_lmax)) - shift[i]
            delta_g    = bg[i]*delta_m
            Ng         = Alm2Map.apply(Map2Alm.apply(self.ng_average[i]*(1.+delta_g), self.lmax)*self.pixwin_ell_filter, self.nside, self.lmax)
            pyro.sample(f'Ng_obs_{i}', dist.Poisson(Ng[self.mask]), obs=self.Ng_obs[i,self.mask])
           
        
    def sample(self, num_burn, num_samples,step_size=0.05, x_init=None,bg_init=None,cosmo_init=None,adapt_mass_mat=True):
        kernel = NUTS(self.model, target_accept_prob=0.65, step_size=step_size)

        x_real_init = 0.3 * torch.randn((self.N_Z_BINS, (self.ell > 1).sum()), dtype=torch.double)
        x_imag_init = 0.3 * torch.randn((self.N_Z_BINS, ((self.ell > 1) & (self.emm > 0)).sum()), dtype=torch.double)
        bg_init     = self.bfid
        cosmo_init  = self.thetafid

        if x_init is not None:
            print('Initialization file found...')
            xlm_real_init, xlm_imag_init = x_init

            bg_init       = torch.tensor(bg_init, dtype=torch.double)
            cosmo_init    = torch.tensor(cosmo_init, dtype=torch.double)
            xlm_real_init = torch.tensor(xlm_real_init, dtype=torch.double)
            xlm_imag_init = torch.tensor(xlm_imag_init, dtype=torch.double)
        
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=num_burn,
                    initial_params={"bg": bg_init,
                                    "cosmo": cosmo_init,
                                    "xlm_imag": x_imag_init,
                                    "xlm_real": x_real_init
                                    })
        mcmc.run()
        self.samps = mcmc.get_samples()
        return self.samps, mcmc.kernel

    def save_samples(self, fname):
        pickle.dump(self.samps, open(fname, 'wb'))
