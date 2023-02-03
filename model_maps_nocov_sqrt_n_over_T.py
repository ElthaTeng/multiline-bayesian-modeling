import time
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from astropy.io import fits

source = 'NGC3627'
start_time = time.time()

two_comp = False
model = '6d_coarse2'
sou_model = 'radex_model/'
sou_data = 'data_image/'

# Set up constraints if using priors 
los_max = 200.
x_co = 3 * 10**(-4)
Xco2alpha = 1 / (4.5 * 10**19) / x_co
map_ew = fits.open(sou_data+source+'_CO10_ew_broad_nyq.fits')[0].data
map_fwhm = map_ew * 2.35  # Conversion of 1 sigma to FWHM assuming Gaussian

map_1dmax = np.full(map_ew.shape, np.nan)
map_neg1sig = np.full(map_ew.shape, np.nan)
map_median = np.full(map_ew.shape, np.nan)
map_pos1sig = np.full(map_ew.shape, np.nan)

# Set parameter ranges
if two_comp:
    samples_Nco = np.arange(16., 19.1, 0.25).astype('float32')
    samples_Tk = np.arange(1.,2.1,0.1).astype('float32')
    samples_nH2 = np.arange(2., 5.1, 0.25).astype('float32')
    samples_phi = np.arange(-1.3, -0.05, 0.1).astype('float32')
else:
    samples_Nco = np.arange(15.,20.1,0.2)
    samples_Tk = np.arange(1.,2.8,0.1)
    samples_nH2 = np.arange(2.,5.1,0.2)
    samples_X12to13 = np.arange(10,205,10)
    samples_X13to18 = np.arange(2,21,1)
    samples_phi = np.arange(-1.3, 0.1, 0.1)

size_N = samples_Nco.shape[0]
size_T = samples_Tk.shape[0]
size_n = samples_nH2.shape[0]
size_phi = samples_phi.shape[0]

if two_comp:
    Nco_0 = samples_Nco.reshape(size_N,1,1,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
    phi_0 = samples_phi.reshape(1,1,1,size_phi,1,1,1,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
    Nco_1 = samples_Nco.reshape(1,1,1,1,size_N,1,1,1)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
    phi_1 = samples_phi.reshape(1,1,1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_phi,size_N,size_T,size_n,size_phi),dtype='float32')
    Nco_0 = Nco_0.reshape(-1)
    phi_0 = phi_0.reshape(-1)
    Nco_1 = Nco_1.reshape(-1)
    phi_1 = phi_1.reshape(-1)

    model_co10 = np.load(sou_model+'fluxsum_'+model+'_co10_8d.npy')   
    model_co21 = np.load(sou_model+'fluxsum_'+model+'_co21_8d.npy')
    model_13co21 = np.load(sou_model+'fluxsum_'+model+'_13co21_8d.npy')
    model_13co32 = np.load(sou_model+'fluxsum_'+model+'_13co32_8d.npy')
    model_c18o21 = np.load(sou_model+'fluxsum_'+model+'_c18o21_8d.npy')
    model_c18o32 = np.load(sou_model+'fluxsum_'+model+'_c18o32_8d.npy')

else:
    size_x12to13 = samples_X12to13.shape[0]
    size_x13to18 = samples_X13to18.shape[0]

    Nco = samples_Nco.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
    Tk = samples_Tk.reshape(1,size_T,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
    nH2 = samples_nH2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
    phi = samples_phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
    Nco = Nco.reshape(-1)
    Tk = Tk.reshape(-1)
    nH2 = nH2.reshape(-1)
    phi = phi.reshape(-1)

    model_co10 = np.load(sou_model+'flux_'+model+'_co10.npy')
    model_co21 = np.load(sou_model+'flux_'+model+'_co21.npy')
    model_13co21 = np.load(sou_model+'flux_'+model+'_13co21.npy')
    model_13co32 = np.load(sou_model+'flux_'+model+'_13co32.npy')
    model_c18o21 = np.load(sou_model+'flux_'+model+'_c18o21.npy')
    model_c18o32 = np.load(sou_model+'flux_'+model+'_c18o32.npy') 

flux_co10 = fits.open(sou_data+source+'_CO10_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_co21 = fits.open(sou_data+source+'_CO21_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_13co21 = fits.open(sou_data+source+'_13CO21_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_13co32 = fits.open(sou_data+source+'_13CO32_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_c18o21 = fits.open(sou_data+source+'_C18O21_mom0_broad_nyq.fits')[0].data.astype('float32')
flux_c18o32 = fits.open(sou_data+source+'_C18O32_mom0_broad_nyq.fits')[0].data.astype('float32')

noise_co10 = fits.open(sou_data+'errors/'+source+'_CO10_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_co21 = fits.open(sou_data+'errors/'+source+'_CO21_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_13co21 = fits.open(sou_data+'errors/'+source+'_13CO21_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_13co32 = fits.open(sou_data+'errors/'+source+'_13CO32_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_c18o21 = fits.open(sou_data+'errors/'+source+'_C18O21_emom0_broad_nyq.fits')[0].data.astype('float32')
noise_c18o32 = fits.open(sou_data+'errors/'+source+'_C18O32_emom0_broad_nyq.fits')[0].data.astype('float32')

err_co10 = np.sqrt(noise_co10**2 + (0.1 * flux_co10)**2)
err_co21 = np.sqrt(noise_co21**2 + (0.1 * flux_co21)**2)
err_13co21 = np.sqrt(noise_13co21**2 + (0.2 * flux_13co21)**2)
err_13co32 = np.sqrt(noise_13co32**2 + (0.2 * flux_13co32)**2)
err_c18o21 = np.sqrt(noise_c18o21**2 + (0.2 * flux_c18o21)**2)
err_c18o32 = np.sqrt(noise_c18o32**2 + (0.2 * flux_c18o32)**2)

# Import data image masks to identify bad pixels with mom0 + emom0 < 0
mask_13co32 = ~np.load(source+'/mask_bad_13co32.npy')
mask_c18o21 = ~np.load(source+'/mask_bad_c18o21.npy')
mask_c18o32 = ~np.load(source+'/mask_bad_c18o32.npy')

data_time = time.time()
print('Models & data loaded;', data_time-start_time, 'sec elapsed.')

def chi2_prob(x,y):
    chi_sum = ((model_co10 - flux_co10[y,x]) / err_co10[y,x])**2
    chi_sum += ((model_co21 - flux_co21[y,x]) / err_co21[y,x])**2
    chi_sum += ((model_13co21 - flux_13co21[y,x]) / err_13co21[y,x])**2

    if mask_13co32[y,x]:
        chi_sum += ((model_13co32 - flux_13co32[y,x]) / err_13co32[y,x])**2
    if mask_c18o21[y,x]:
        chi_sum += ((model_c18o21 - flux_c18o21[y,x]) / err_c18o21[y,x])**2
    if mask_c18o32[y,x]:
        chi_sum += ((model_c18o32 - flux_c18o32[y,x]) / err_c18o32[y,x])**2
    
    prob = np.nan_to_num(np.exp(-0.5 * chi_sum)).reshape(-1)   
    return chi_sum.reshape(-1), prob

def prob1d_maps(x,y):  
    chi2, prob = chi2_prob(x,y)
    if two_comp == False:
        los_length = (10**Nco / 15. * map_fwhm[y,x]) / (np.sqrt(10**phi) * 10**nH2 * x_co)
        mask = los_length < los_max * (3.086 * 10**18) 
        chi2 = np.array(chi2) * mask
        chi2[mask==0] = np.nan
        prob = np.array(prob) * mask
    
    if (np.isnan(chi2)).all():
        return x, y, np.nan, np.nan, np.nan, np.nan

    else:
        if two_comp:
            grid = np.load('radex_model/tau_'+model+'_co10.npy').reshape(-1)
        else:
            grid = nH2 / 2 - Tk
            grid[mask == 0] = np.nan

        num_bins = 20
        counts_noweight, bins = np.histogram(grid, bins=num_bins, range=(-2.,2.), weights=None, density=True)
        counts_weighted, bins = np.histogram(grid, bins=num_bins, range=(-2.,2.), weights=prob)
        counts_norm = np.nan_to_num(counts_weighted / counts_noweight)
        samples_grid = bins[:-1]

        ## Save the whole 1D likelihood distribution
        np.save(sou_model+source+'_prob1d/sqrt_n_over_T_'+model+'_'+str(x)+'_'+str(y)+'.npy', counts_norm)
        ## 1DMax
        out_1dmax = samples_grid[np.nanargmax(counts_norm)]
        ## CDF percentiles
        quantile_values = np.array((0.16,0.5,0.84),dtype='float32')
        cdf = np.cumsum(counts_norm)
        cdf /= cdf[-1]
        out_neg1sig = np.interp(quantile_values[0], cdf, samples_grid) 
        out_median = np.interp(quantile_values[1], cdf, samples_grid)
        out_pos1sig = np.interp(quantile_values[2], cdf, samples_grid)
        #out_value = corner.quantile(alpha, [0.16,0.5,0.84], weights=prob)[2] 

        return x, y, out_1dmax, out_neg1sig, out_median, out_pos1sig 

print('Start multi-processing on output maps')
 
if source == 'NGC4321':
    results = Parallel(n_jobs=10, verbose=10)(delayed(prob1d_maps)(x,y) for x in range(35,90) for y in range(35,90)) 
elif source == 'NGC3627':
    results = Parallel(n_jobs=10, verbose=10)(delayed(prob1d_maps)(x,y) for x in range(30,75) for y in range(30,75))
     
for result in results:
    x, y, out_1dmax, out_neg1sig, out_median, out_pos1sig = result 
    map_1dmax[y,x] = out_1dmax
    map_neg1sig[y,x] = out_neg1sig
    map_median[y,x] = out_median
    map_pos1sig[y,x] = out_pos1sig

    np.save(sou_model+'sqrt_n_over_T_'+model+'_'+source+'_1dmax_los200.npy', map_1dmax)
    np.save(sou_model+'sqrt_n_over_T_'+model+'_'+source+'_neg1sig_los200.npy', map_neg1sig)
    np.save(sou_model+'sqrt_n_over_T_'+model+'_'+source+'_median_los200.npy', map_median)
    np.save(sou_model+'sqrt_n_over_T_'+model+'_'+source+'_pos1sig_los200.npy', map_pos1sig)
    
print('Output maps constructed.')

end_time = time.time()
print('Total elapsed time: %s sec' % ((end_time - start_time)))
