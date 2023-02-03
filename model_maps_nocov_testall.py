import time
import numpy as np
from joblib import Parallel, delayed
import corner
from astropy.io import fits

start_time = time.time()

source = 'NGC4321'
test_constraint = 'rm13co32'
output_map = 'median_interp'
sou_data = 'data_image/'

if source=='NGC3351':
    model = '6d_coarse'
    sou_model = 'radex_model/ngc3351_npy/'
    los_max = 100.

    samples_Nco = np.arange(16.,21.1,0.2)
    samples_Tk = np.arange(1.,2.4,0.1)
    samples_X13to18 = np.arange(2,21,1.5)
    samples_phi = np.arange(0.05, 1.01, 0.05)
else:
    model = '6d_coarse2'
    sou_model = 'radex_model/'
    los_max = 200.

    samples_Nco = np.arange(15.,20.1,0.2)
    samples_Tk = np.arange(1.,2.8,0.1)
    samples_X13to18 = np.arange(2,21,1)
    samples_phi = np.arange(-1.3, 0.1, 0.1)

samples_nH2 = np.arange(2.,5.1,0.2)
samples_X12to13 = np.arange(10,205,10)
x_co = 3 * 10**(-4)

map_ew = fits.open('data_image/'+source+'_CO21_ew_broad_nyq.fits')[0].data
map_fwhm = map_ew * 2.35  # Conversion of 1 sigma to FWHM assuming Gaussian

chi2_map = np.full(map_ew.shape, np.nan)
N_co = np.full(map_ew.shape, np.nan)
T_k = np.full(map_ew.shape, np.nan)
n_h2 = np.full(map_ew.shape, np.nan)
X_co213co = np.full(map_ew.shape, np.nan)
X_13co2c18o = np.full(map_ew.shape, np.nan)
beam_fill = np.full(map_ew.shape, np.nan)

size_N = samples_Nco.shape[0]
size_T = samples_Tk.shape[0]
size_n = samples_nH2.shape[0]
size_x12to13 = samples_X12to13.shape[0]
size_x13to18 = samples_X13to18.shape[0]
size_phi = samples_phi.shape[0]

Nco = samples_Nco.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
Nco = Nco.reshape(-1)
Tk = samples_Tk.reshape(1,size_T,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
Tk = Tk.reshape(-1)
nH2 = samples_nH2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
nH2 = nH2.reshape(-1)
X12to13 = samples_X12to13.reshape(1,1,1,size_x12to13,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
X12to13 = X12to13.reshape(-1)
X13to18 = samples_X13to18.reshape(1,1,1,1,size_x13to18,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
X13to18 = X13to18.reshape(-1)
phi = samples_phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
phi = phi.reshape(-1)

model_co10 = np.load(sou_model+'flux_'+model+'_co10.npy')    
model_co21 = np.load(sou_model+'flux_'+model+'_co21.npy')
model_13co21 = np.load(sou_model+'flux_'+model+'_13co21.npy')
model_13co32 = np.load(sou_model+'flux_'+model+'_13co32.npy')

flux_co10 = fits.open(sou_data+source+'_CO10_mom0_broad_nyq.fits')[0].data
flux_co21 = fits.open(sou_data+source+'_CO21_mom0_broad_nyq.fits')[0].data
flux_13co21 = fits.open(sou_data+source+'_13CO21_mom0_broad_nyq.fits')[0].data
flux_13co32 = fits.open(sou_data+source+'_13CO32_mom0_broad_nyq.fits')[0].data

noise_co10 = fits.open(sou_data+'errors/'+source+'_CO10_emom0_broad_nyq.fits')[0].data
noise_co21 = fits.open(sou_data+'errors/'+source+'_CO21_emom0_broad_nyq.fits')[0].data
noise_13co21 = fits.open(sou_data+'errors/'+source+'_13CO21_emom0_broad_nyq.fits')[0].data
noise_13co32 = fits.open(sou_data+'errors/'+source+'_13CO32_emom0_broad_nyq.fits')[0].data

err_co10 = np.sqrt(noise_co10**2 + (0.1 * flux_co10)**2)
err_co21 = np.sqrt(noise_co21**2 + (0.1 * flux_co21)**2)
if source=='NGC3351':
    err_13co21 = np.sqrt(noise_13co21**2 + (0.1 * flux_13co21)**2)
    err_13co32 = np.sqrt(noise_13co32**2 + (0.1 * flux_13co32)**2)
else:
    err_13co21 = np.sqrt(noise_13co21**2 + (0.2 * flux_13co21)**2)
    err_13co32 = np.sqrt(noise_13co32**2 + (0.2 * flux_13co32)**2)

data_time = time.time()
print('Models & data loaded;', data_time-start_time, 'sec elapsed.')

def chi2_prob(x,y):
    chi_sum = ((model_co21 - flux_co21[y,x]) / err_co21[y,x])**2
    chi_sum += ((model_13co21 - flux_13co21[y,x]) / err_13co21[y,x])**2

    if test_constraint=='rmco10':    
        chi_sum += ((model_13co32 - flux_13co32[y,x]) / err_13co32[y,x])**2
    elif test_constraint=='rm13co32':
        chi_sum += ((model_co10 - flux_co10[y,x]) / err_co10[y,x])**2
    elif test_constraint=='all4':
        chi_sum += ((model_co10 - flux_co10[y,x]) / err_co10[y,x])**2
        chi_sum += ((model_13co32 - flux_13co32[y,x]) / err_13co32[y,x])**2
    
    prob = np.nan_to_num(np.exp(-0.5 * chi_sum)).reshape(-1)   
    return chi_sum.reshape(-1), prob

def bestfit_maps(x,y):   
    if source=='NGC3351':
        los_length = (10**Nco / 15. * map_fwhm[y,x]) / (np.sqrt(phi) * 10**nH2 * x_co)
    else:
        los_length = (10**Nco / 15. * map_fwhm[y,x]) / (np.sqrt(10**phi) * 10**nH2 * x_co)
    mask = los_length < los_max * (3.086 * 10**18) 
    chi2, _ = chi2_prob(x,y)
    chi2 = np.array(chi2) * mask
    chi2[mask==0] = np.nan
    
    if (np.isnan(chi2)).all():
        return x, y, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    else:
        chi2_min = np.nanmin(chi2)
        idx_min = np.unravel_index(np.nanargmin(chi2), (size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
        N_value = idx_min[0] * (samples_Nco[1] - samples_Nco[0]) + samples_Nco[0]
        T_value = idx_min[1] * (samples_Tk[1] - samples_Tk[0]) + samples_Tk[0]
        n_value = idx_min[2] * (samples_nH2[1] - samples_nH2[0]) + samples_nH2[0]
        X1_value = idx_min[3] * (samples_X12to13[1] - samples_X12to13[0]) + samples_X12to13[0]
        X2_value = idx_min[4] * (samples_X13to18[1] - samples_X13to18[0]) + samples_X13to18[0]
        phi_value = idx_min[5] * (samples_phi[1] - samples_phi[0]) + samples_phi[0]
        return x, y, chi2_min, N_value, T_value, n_value, X1_value, X2_value, phi_value 

def prob1d_maps(x,y):  
    if source=='NGC3351':
        los_length = (10**Nco / 15. * map_fwhm[y,x]) / (np.sqrt(phi) * 10**nH2 * x_co)
    else:
        los_length = (10**Nco / 15. * map_fwhm[y,x]) / (np.sqrt(10**phi) * 10**nH2 * x_co)
    mask = los_length < los_max * (3.086 * 10**18) 
    chi2, prob = chi2_prob(x,y)
    chi2 = np.array(chi2) * mask
    chi2[mask==0] = np.nan
    prob = np.array(prob) * mask
    
    if (np.isnan(chi2)).all():
        return x, y, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    elif output_map == 'mean':
        N_value = np.sum(Nco * prob) / prob.sum()
        T_value = np.sum(Tk * prob) / prob.sum()
        n_value = np.sum(nH2 * prob) / prob.sum()
        X1_value = np.sum(X12to13 * prob) / prob.sum()
        X2_value = np.sum(X13to18 * prob) / prob.sum() 
        phi_value = np.sum(phi * prob) / prob.sum()
        return x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value  

    elif output_map == 'median':
        N_value = corner.quantile(Nco, [0.16,0.5,0.84], weights=prob)[2]  #0:neg1sig, 1:median, 2:pos1sig
        T_value = corner.quantile(Tk, [0.16,0.5,0.84], weights=prob)[2]
        n_value = corner.quantile(nH2, [0.16,0.5,0.84], weights=prob)[2]
        X1_value = corner.quantile(X12to13, [0.16,0.5,0.84], weights=prob)[2]
        X2_value = corner.quantile(X13to18, [0.16,0.5,0.84], weights=prob)[2]
        phi_value = corner.quantile(phi, [0.16,0.5,0.84], weights=prob)[2]
        return x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value 
        
    else:
        n_1d, _ = np.histogram(nH2, bins=size_n, range=(1.9,5.1), weights=prob) 
        X1_1d, _ = np.histogram(X12to13, bins=size_x12to13, range=(5,205), weights=prob) 

        if source=='NGC3351':
            N_1d, _ = np.histogram(Nco, bins=size_N, range=(15.9,21.1), weights=prob) 
            T_1d, _ = np.histogram(Tk, bins=size_T, range=(0.95,2.35), weights=prob) 
            X2_1d, _ = np.histogram(X13to18, bins=size_x13to18, range=(1.25,20.75), weights=prob) 
            phi_1d, _ = np.histogram(phi, bins=size_phi, range=(0.025,1.025), weights=prob)
        else:
            N_1d, _ = np.histogram(Nco, bins=size_N, range=(14.9,20.1), weights=prob) 
            T_1d, _ = np.histogram(Tk, bins=size_T, range=(0.95,2.75), weights=prob) 
            X2_1d, _ = np.histogram(X13to18, bins=size_x13to18, range=(1.5,20.5), weights=prob) 
            phi_1d, _ = np.histogram(phi, bins=size_phi, range=(-1.35,0.05), weights=prob)
        
        if output_map == '1dmax':      
            N_value = samples_Nco[np.nanargmax(N_1d)]
            T_value = samples_Tk[np.nanargmax(T_1d)]
            n_value = samples_nH2[np.nanargmax(n_1d)]
            X1_value = samples_X12to13[np.nanargmax(X1_1d)]
            X2_value = samples_X13to18[np.nanargmax(X2_1d)]
            phi_value = samples_phi[np.nanargmax(phi_1d)]
            return x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value
            
        elif output_map == 'median_interp':
            quantile_values = np.array((0.16,0.5,0.84),dtype='float32')
            cdf_N = np.cumsum(N_1d)
            cdf_N /= cdf_N[-1]
            cdf_T = np.cumsum(T_1d)
            cdf_T /= cdf_T[-1]
            cdf_n = np.cumsum(n_1d)
            cdf_n /= cdf_n[-1]
            cdf_X1 = np.cumsum(X1_1d)
            cdf_X1 /= cdf_X1[-1]
            cdf_X2 = np.cumsum(X2_1d)
            cdf_X2 /= cdf_X2[-1]
            cdf_phi = np.cumsum(phi_1d)
            cdf_phi /= cdf_phi[-1]          

            N_value = np.interp(quantile_values[1], cdf_N, samples_Nco)
            T_value = np.interp(quantile_values[1], cdf_T, samples_Tk)
            n_value = np.interp(quantile_values[1], cdf_n, samples_nH2)
            X1_value = np.interp(quantile_values[1], cdf_X1, samples_X12to13)
            X2_value = np.interp(quantile_values[1], cdf_X2, samples_X13to18)
            phi_value = np.interp(quantile_values[1], cdf_phi, samples_phi)
            return x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value 

print('Start multi-processing on output maps')
 
if output_map == 'bestfit':
    if source == 'NGC4321':
        results = Parallel(n_jobs=10, verbose=5)(delayed(bestfit_maps)(x,y) for x in range(35,90) for y in range(35,90))
    elif source == 'NGC3627':
        results = Parallel(n_jobs=10, verbose=5)(delayed(bestfit_maps)(x,y) for x in range(30,75) for y in range(30,75))   
    elif source == 'NGC3351':
        results = Parallel(n_jobs=10, verbose=5)(delayed(bestfit_maps)(x,y) for x in range(20,55) for y in range(20,55))  

    for result in results:
        x, y, chi2_min, N_value, T_value, n_value, X1_value, X2_value, phi_value = result
        chi2_map[y,x] = chi2_min
        N_co[y,x] = N_value
        T_k[y,x] = T_value
        n_h2[y,x] = n_value
        X_co213co[y,x] = X1_value
        X_13co2c18o[y,x] = X2_value
        beam_fill[y,x] = phi_value
        
    np.save(sou_model+'chi2_'+model+'_rmcor_'+source+'_'+test_constraint+'_los200.npy', chi2_map)
    np.save(sou_model+'Nco_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', N_co)
    np.save(sou_model+'Tk_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', T_k)
    np.save(sou_model+'nH2_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', n_h2)
    np.save(sou_model+'X12to13_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', X_co213co)
    np.save(sou_model+'X13to18_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', X_13co2c18o)
    np.save(sou_model+'phi_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', beam_fill)
    
else:
    if source == 'NGC4321':
        results = Parallel(n_jobs=10, verbose=5)(delayed(prob1d_maps)(x,y) for x in range(35,90) for y in range(35,90)) 
    elif source == 'NGC3627':
        results = Parallel(n_jobs=10, verbose=5)(delayed(prob1d_maps)(x,y) for x in range(30,75) for y in range(30,75))   
    elif source == 'NGC3351':
        results = Parallel(n_jobs=10, verbose=5)(delayed(prob1d_maps)(x,y) for x in range(20,55) for y in range(20,55))  

    for result in results:
        x, y, N_value, T_value, n_value, X1_value, X2_value, phi_value = result
        N_co[y,x] = N_value
        T_k[y,x] = T_value
        n_h2[y,x] = n_value
        X_co213co[y,x] = X1_value
        X_13co2c18o[y,x] = X2_value
        beam_fill[y,x] = phi_value        
    
    np.save(sou_model+'Nco_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', N_co)
    np.save(sou_model+'Tk_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', T_k)
    np.save(sou_model+'nH2_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', n_h2)
    np.save(sou_model+'X12to13_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', X_co213co)
    np.save(sou_model+'X13to18_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', X_13co2c18o)
    np.save(sou_model+'phi_'+model+'_rmcor_'+source+'_los200_'+output_map+'_'+test_constraint+'.npy', beam_fill)
    
print('Output maps constructed.')

end_time = time.time()
print('Total elapsed time: %s sec' % ((end_time - start_time)))