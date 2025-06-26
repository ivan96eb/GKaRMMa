import numpy as np
import pyccl as ccl
from scipy.special import eval_legendre
from joblib import Parallel, delayed
import healpy as hp
from scipy.stats import qmc
import h5py as h5
def normal_dist(x,mu,sigma):

    """ Returns the value of the gaussian
    PDF with parameters mu and sigma
    at a given point x.

    Parameters
    ----------
    x : float
        Point
    mu : float
        mean of the distribution
    sigma : float
        Standard deviation of the 
        distribution
    """

    return np.exp(-((x-mu)/sigma)**2)

def powerspectrum_generator(nbins,z,n,ell,omega_m_fid,sigma_8_fid,cross_corr):

    cl = np.zeros((nbins,nbins,len(ell)))

    cosmo = ccl.Cosmology(Omega_c=omega_m_fid, Omega_b=0.05,
                            h=0.7, n_s=0.95, sigma8=sigma_8_fid,
                            transfer_function='bbks')
    if cross_corr==True:
        print('working with cross correlation')

        for i in range(nbins):
            for j in range(nbins):
                gals_i = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, n[i]),
                                                bias=(z, np.ones_like(z)))
                gals_j = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, n[j]),
                                                bias=(z, np.ones_like(z)))
                cl[i,j] = ccl.angular_cl(cosmo, gals_i, gals_j, ell)
    else: 
        print('working with NO cross correlation')
        for i in range(nbins):
            gals = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, n[i]),
                                        bias=(z, np.ones_like(z)))
            cl[i,i] = ccl.angular_cl(cosmo, gals, gals, ell)
    return cl

def compute_lognorm_cl_at_ell(mu, w, integrand, ell):
    xi_g = np.log(np.polynomial.legendre.legval(mu, integrand) + 1)
    return 2 * np.pi * np.sum(w * xi_g * eval_legendre(ell, mu))

def compute_lognorm_cl(cl,nside,nbins,shift,vargauss,order=2):
    gen_lmax = 3*nside-1
    mu, w = np.polynomial.legendre.leggauss(order * gen_lmax)
    gauss_mu = np.zeros(nbins)
    for i in range(nbins):           
        gauss_mu[i] = np.log(shift[i]) - 0.5 * vargauss[i]            
    y_cl = np.zeros_like(cl)
    print("Computing y_cl...")
    for i in range(nbins):    
        for j in range(i+1):
            print("z-bin i: %d, j: %d"%(i,j))
            integrand = ((2 * np.arange(gen_lmax + 1) + 1) * cl[i,j] / (4 * np.pi * shift[i] * shift[j]))
            ycl_ij = np.array(Parallel(n_jobs=-1)(
        delayed(compute_lognorm_cl_at_ell)(mu, w, integrand, ell) for ell in range(gen_lmax + 1)))
            y_cl[i,j] = ycl_ij
            y_cl[j,i] = ycl_ij
            
    y_cl[:,:,:2]  = np.tile(1e-20 * np.eye(nbins)[:,:,np.newaxis], (1,1,2))
    return y_cl


def LHC(omega_min,omega_max,sigma_min,sigma_max,Nsamps):
    sampler  = qmc.LatinHypercube(d=2)
    sample   = sampler.random(n=Nsamps)
    l_bounds = [omega_min,sigma_min]
    u_bounds = [omega_max,sigma_max]

    return qmc.scale(sample, l_bounds, u_bounds)

def eigvec_matmul(A, x, nbins):
    y = np.zeros_like(x)
    for i in range(nbins):
        for j in range(nbins):
            y[i] += A[i,j] * x[j]
    return y

def apply_cl(xlm, cl,nbins,gen_lmax):
    ell, emm = hp.Alm.getlm(gen_lmax)
    L        = np.linalg.cholesky(cl.T).T
    
    xlm_real = xlm.real
    xlm_imag = xlm.imag
    
    L_arr = np.swapaxes(L[:,:,ell[ell > -1]], 0,1)
    
    ylm_real = eigvec_matmul(L_arr, xlm_real,nbins) / np.sqrt(2.)
    ylm_imag = eigvec_matmul(L_arr, xlm_imag,nbins) / np.sqrt(2.)

    ylm_real[:,ell[emm==0]] *= np.sqrt(2)
    
    return ylm_real + 1j * ylm_imag

def get_xlm(xlm_real, xlm_imag, nbins, gen_lmax):
    ell, emm = hp.Alm.getlm(gen_lmax)
    #==============================
    _xlm_real = np.zeros((nbins, len(ell)))
    _xlm_imag = np.zeros_like(_xlm_real)

    _xlm_real[:,ell > 1]               = xlm_real
    _xlm_imag[:,(ell > 1) & (emm > 0)] = xlm_imag
    xlm = _xlm_real + 1j * _xlm_imag
    #==============================
    return xlm
    
def generate_xlm(nbins,gen_lmax):
    ell, emm = hp.Alm.getlm(gen_lmax)
    xlm_real = np.random.normal(size=(nbins, (ell > 1).sum()))
    xlm_imag = np.random.normal(size=(nbins, ((ell > 1) & (emm > 0)).sum()))
    xlm      = get_xlm(xlm_real, xlm_imag, nbins, gen_lmax)
    return xlm, [xlm_real,xlm_imag]

def generate_mock_y_lm(ycl,nbins,gen_lmax):
    xlm, _xlms = generate_xlm(nbins,gen_lmax)
    return apply_cl(xlm, ycl,nbins,gen_lmax),_xlms

def get_GRF(ycl,nside,nbins,gen_lmax):
    y_lm,_xlms   = generate_mock_y_lm(ycl,nbins,gen_lmax)
    y_maps = []
    for i in range(nbins):
        y_map = hp.alm2map(np.ascontiguousarray(y_lm[i]), nside, lmax=gen_lmax, pol=False)
        y_maps.append(y_map)    
    return np.array(y_maps),_xlms    

def get_LNRF(y,shift,mu):
    delta_g = np.zeros_like(y)
    nbins   = y.shape[0]
    for i in range(nbins):
        delta_g[i] = np.exp(y[i]+mu[i]) - shift[i]
    return delta_g

def apply_bias(delta_m,b):
    nbins   = delta_m.shape[0]
    delta_g = np.zeros_like(delta_m)
    for i in range(nbins):
        delta_g[i] = b[i]*delta_m[i]
    return delta_g 

def low_pass_filter(map):
    nbins    = map.shape[0]
    nside    = hp.npix2nside(map.shape[1])
    gen_lmax = 3*nside - 1
    lmax     = 2*nside
    map_lp   = np.zeros_like(map)
    ell,_    = hp.Alm.getlm(gen_lmax)
    for i in range(nbins):
        map_lm           = hp.map2alm(map[i],lmax=gen_lmax)
        map_lm[ell>lmax] = 0.+0.*1j
        map_lp[i]        = hp.alm2map(map_lm,nside=nside)
    return map_lp

def get_Ng(delta_g,N_gals_average,boolean_mask):
    nbins = delta_g.shape[0]
    Ng    = np.zeros_like(delta_g)
    for i in range(nbins):
        Ng[i][boolean_mask] = np.random.poisson(N_gals_average[i]*(1.+delta_g[i][boolean_mask]))
    return Ng

def pixelize(map,pixwin):
    nbins  = map.shape[0]
    nside  = hp.npix2nside(map.shape[1])
    lmax   = 2*nside
    pixmap = np.zeros_like(map)
    for i in range(nbins):
        alm       = hp.map2alm(map[i],lmax=lmax)
        alm_pix   = alm*pixwin
        pixmap[i] = hp.alm2map(alm_pix,nside)
    return pixmap

def save_datafile(N_gals,mask,delta_g,cosmo,bg,xlm,outpath):
    hf = h5.File(outpath, 'w')
    hf.create_dataset('mask',     data = mask)
    hf.create_dataset('Ng_obs',   data = N_gals)
    hf.create_dataset('delta_g',  data = delta_g)
    hf.create_dataset('cosmo',    data = cosmo)
    hf.create_dataset('bg',       data = bg)
    hf.create_dataset('xlm_real', data = xlm[0])
    hf.create_dataset('xlm_imag', data = xlm[1])
    hf.close()