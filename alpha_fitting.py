import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

fit_option = 'tauT' #'ew' #'ratio21' #

def func_doublevar(X, a, b, c):
    logtau, logT = X
    return a * logtau + b * logT + c

def func_singlevar(X, a, b):
    return a * X + b 

def map_masked(source, map_type):
    if source == 'NGC3351':
        mask = (np.load(source+'/mask_whole_recovered.npy') * np.load(source+'/mask_cent3sig.npy') * np.load(source+'/mask_rmcor_comb_lowchi2.npy') 
            * np.isfinite(np.log10(np.load(source+'/data_image/ratio_CO21_13CO21_broad.npy'))) * (np.load(source+'/radex_model/tau_6d_coarse2_'+source+'_co21_median_los200.npy') > 0.7))
        if map_type == 'alpha':
            map = np.load(source+'/radex_model/Xco_6d_coarse_ewsame_median_los100.npy') 
        elif map_type == 'tau':
            map = np.load(source+'/radex_model/tau_6d_coarse_co21_median_los100.npy')
        elif map_type == 'Tk':
            map = np.load(source+'/radex_model/Tk_6d_coarse_rmcor_whole_los100_median_interp.npy') 
    else:
        mask = (np.load(source+'/mask_13co21_3sig.npy') * np.load(source+'/mask_recovered_0.3.npy') 
            * (np.load(source+'/data_image/'+source+'_CO21_mom0.npy') > 50) * (np.load(source+'/radex_model/tau_6d_coarse2_'+source+'_co21_median_los200.npy') > 0.7))
        if map_type == 'alpha':
            map = np.load(source+'/radex_model/Xco_6d_coarse2_'+source+'_ewsame_median_los200.npy') 
        elif map_type == 'tau':
            map = np.load(source+'/radex_model/tau_6d_coarse2_'+source+'_co21_median_los200.npy')
        elif map_type == 'Tk':
            map = np.load(source+'/radex_model/Tk_6d_coarse2_rmcor_'+source+'_los200_median_interp.npy') 

    if map_type == 'ew':
        map = np.log10(fits.open(source+'/data_image/'+source+'_CO21_ew_broad_nyq.fits')[0].data)
    elif map_type == 'ratio21':
        map = np.log10(np.load(source+'/data_image/ratio_CO21_13CO21.npy')) 
        
    out_map = map * mask
    out_map[mask==0] = np.nan
    
    return out_map.reshape(-1)

alpha = np.concatenate((np.concatenate((map_masked('NGC3351', 'alpha'), map_masked('NGC3627', 'alpha')), axis=None), map_masked('NGC4321', 'alpha')), axis=None) 
tau = np.concatenate((np.concatenate((map_masked('NGC3351', 'tau'), map_masked('NGC3627', 'tau')), axis=None), map_masked('NGC4321', 'tau')), axis=None) 
Tk = np.concatenate((np.concatenate((map_masked('NGC3351', 'Tk'), map_masked('NGC3627', 'Tk')), axis=None), map_masked('NGC4321', 'Tk')), axis=None) 
ew = np.concatenate((np.concatenate((map_masked('NGC3351', 'ew'), map_masked('NGC3627', 'ew')), axis=None), map_masked('NGC4321', 'ew')), axis=None) 
ratio21 = np.concatenate((np.concatenate((map_masked('NGC3351', 'ratio21'), map_masked('NGC3627', 'ratio21')), axis=None), map_masked('NGC4321', 'ratio21')), axis=None) 

valid = ~(np.isnan(alpha) | np.isnan(ratio21)) # 
N_samples = alpha[valid].shape[0]
print('Sample size:', N_samples)

if fit_option == 'tauT':
    refits = np.full((1000, 6), np.nan)
    ic = [1., -0.6, 0.]  
    popt, pcov = curve_fit(func_doublevar, (tau[valid], Tk[valid]), alpha[valid], p0 = ic)  
else:
    refits = np.full((1000, 4), np.nan)
    ic = [-2, -1]
    if fit_option == 'ew':
        popt, pcov = curve_fit(func_singlevar, ew[valid], alpha[valid], p0 = ic)
    elif fit_option == 'ratio21':    
        popt, pcov = curve_fit(func_singlevar, ratio21[valid], alpha[valid], p0 = ic)
errors = np.sqrt(np.diagonal(pcov))
print('Fitted coefficients:', popt) 
print('Errors:', errors)

# Bootstrap / Resample
for i in range(1000):
    idx_resample = np.random.choice(N_samples, N_samples)
    alpha_resample = alpha[valid][idx_resample]
    
    if fit_option == 'tauT':
        Tk_resample = Tk[valid][idx_resample]
        tau_resample = tau[valid][idx_resample]
        popt, pcov = curve_fit(func_doublevar, (tau_resample, Tk_resample), alpha_resample, p0 = ic)
        errors = np.sqrt(np.diagonal(pcov))
        refits[i,:3] = popt
        refits[i,3:] = errors
    else:
        if fit_option == 'ew':
            ew_resample = ew[valid][idx_resample]
            popt, pcov = curve_fit(func_singlevar, ew[valid], alpha[valid], p0 = ic)
        elif fit_option == 'ratio21':    
            ew_resample = ratio21[valid][idx_resample]
            popt, pcov = curve_fit(func_singlevar, ratio21[valid], alpha[valid], p0 = ic)
        errors = np.sqrt(np.diagonal(pcov))
        refits[i,:2] = popt
        refits[i,2:] = errors   
    
np.save('alpha_refit1000_'+fit_option+'_taugtr5.npy', refits)
print('Mean of 1000 Refits =', np.mean(refits, axis=0))
print('Std of 1000 Refits =', np.std(refits, axis=0))

if fit_option == 'tauT':
    print(np.percentile(func_doublevar((tau[valid], Tk[valid]), popt[0], popt[1], popt[2]) - alpha[valid], 25))
    print(np.percentile(func_doublevar((tau[valid], Tk[valid]), popt[0], popt[1], popt[2]) - alpha[valid], 75))
else:
    plt.scatter(ew[valid], alpha[valid], marker='.', c='gray')
    plt.plot(np.arange(np.nanmin(ew), np.nanmax(ew), 0.1), func_singlevar(np.arange(np.nanmin(ew), np.nanmax(ew), 0.1), popt[0], popt[1]), 'k--')
    plt.show()
    print(np.std(func_singlevar(ew[valid], popt[0], popt[1]) - alpha[valid]))
