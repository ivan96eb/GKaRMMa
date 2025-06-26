import numpy as np
import h5py as h5
import pickle
import yaml
import os

class KarmmaConfig:
    def __init__(self, configfile):
        with open(configfile, "r") as stream:
            config_args = yaml.safe_load(stream)
        self.analysis = self.set_config_analysis(config_args['analysis'])
        self.set_config_io(config_args['io'])
        self.set_config_mcmc(config_args['mcmc'])
        self.set_config_mocks(config_args['mocks'])

    def set_config_analysis(self, config_args_analysis):
        print("Setting config data....")
        nbins            = int(config_args_analysis['nbins'])
        nside            = int(config_args_analysis['nside'])
        split_ng_average = config_args_analysis['ng_average'].split(',')
        ng_average       = np.array([float(split_ng_average[i]) for i in range(nbins)])
        N_theta          = config_args_analysis['N_theta']
        N_bias           = config_args_analysis['N_bias']
        split_theta      = config_args_analysis['theta_fid'].split(',')
        thetafid         = np.array([float(split_theta[i]) for i in range(N_theta)])
        split_bias       = config_args_analysis['b_fid'].split(',')
        bfid             = np.array([float(split_bias[i]) for i in range(N_bias)])
        try:
            pixwin = np.load(config_args_analysis['pixwin'])
            print("USING EMPIRICAL WINDOW FUNCTION!")
        except:
            pixwin='healpix'

        data_dict = {'nbins': nbins, 
                     'nside': nside, 
                     'pixwin': pixwin,
                     'ng_average': ng_average,
                     'thetafid' : thetafid,
                     'bfid' : bfid
                    }

        return data_dict
    
    def set_config_io(self, config_args_io):
        self.datafile = config_args_io['datafile']
        try:
            self.data     = self.read_data(self.datafile)
        except:
            if not os.path.exists(self.datafile):
                print("DATAFILE NOT FOUND!")            
            else:
                print("Error while reading datafile!")
                raise
        self.io_dir   = config_args_io['io_dir']
        try:
            self.maskfile = config_args_io['maskfile']
        except:
            self.maskfile = None
        try:
            with h5.File(config_args_io['x_init_file'], 'r') as f:
                xlm_imag_init   = f['xlm_imag'][:]
                xlm_real_init   = f['xlm_real'][:]
                self.x_init     = [xlm_real_init, xlm_imag_init]
                self.bg_init    = f['bg'][:].clone().detach()
                self.cosmo_init = f['cosmo'][:].clone().detach()
            print("Initialized from file: "+config_args_io['x_init_file'])
        except:
            print("Initialization file not found. Initializing with prior.")
            self.x_init     = None
            self.bg_init    = None
            self.cosmo_init = None
    def read_data(self, datafile):
        with h5.File(datafile, 'r') as f:
            Ng_obs      = f['Ng_obs'][:]
            mask        = f['mask'][:]
        return {'mask': mask,
                'Ng_obs': Ng_obs}
    
    def set_config_mcmc(self, config_args_mcmc):
        self.n_burn_in                   = config_args_mcmc['n_burn_in']
        self.n_samples                   = config_args_mcmc['n_samples']
        self.vargauss_training_data_path = config_args_mcmc['vargauss_training_data_path']
        self.y_cl_training_data_path     = config_args_mcmc['y_cl_training_data_path']
        try:
            self.step_size = float(config_args_mcmc['step_size'])
        except:
            self.step_size = 0.05

    def set_config_mocks(self,config_args_mocks):
        self.N_theta  = config_args_mocks['N_theta']
        self.N_bias   = config_args_mocks['N_bias']
        split_theta   = config_args_mocks['theta_fid'].split(',')
        self.thetafid = np.array([float(split_theta[i]) for i in range(self.N_theta)])
        split_bias    = config_args_mocks['b_fid'].split(',')
        self.bfid     = np.array([float(split_bias[i]) for i in range(self.N_bias)])