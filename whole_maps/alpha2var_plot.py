import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

def as2kpc(x):
    factor = dist * 10**6 / 206265 / 1000
    return x * factor
    
def kpc2as(x):
    factor = dist * 10**6 / 206265 / 1000
    return x / factor

source = 'NGC3627'
dist = 11.32  #9.96, 11.32, 15.21
input = 'radius'
regional = True
compact = True
model = '6d_coarse2'

data_alpha_peak = np.load(source+'/radex_model/Xco_'+model+'_'+source+'_ewsame_1dmax_los200.npy').reshape(-1)
data_alpha_50 = np.load(source+'/radex_model/Xco_'+model+'_'+source+'_ewsame_median_los200.npy').reshape(-1)
data_alpha_84 = np.load(source+'/radex_model/Xco_'+model+'_'+source+'_ewsame_pos1sig_los200.npy').reshape(-1)
data_alpha_16 = np.load(source+'/radex_model/Xco_'+model+'_'+source+'_ewsame_neg1sig_los200.npy').reshape(-1)

mask_whole = np.load(source+'/mask_13co21_3sig.npy') * np.load(source+'/mask_recovered_0.3.npy') * (np.load(source+'/data_image/'+source+'_CO21_mom0.npy') > 50) 
if regional:
    mask = mask_whole * np.load(source+'/mask_inflow.npy') 
mask = mask.reshape(-1)

if input == 'radius':
    var = np.load(source+'/'+source+'_radius_arcsec.npy').reshape(-1)
elif input == 'tau':
    var = np.load(source+'/radex_model/tau_'+model+'_'+source+'_co10_median_los200.npy').reshape(-1)
    Nco = (np.load(source+'/radex_model/Nco_'+model+'_rmcor_'+source+'_los200_median_interp.npy')
            + np.log10(fits.open(source+'/data_image/'+source+'_CO21_ew_broad_nyq.fits')[0].data / 15)).reshape(-1)
elif input == 'Tk': 
    var = np.load(source+'/radex_model/Tk_'+model+'_rmcor_'+source+'_los200_median_interp.npy').reshape(-1)
elif input == 'ratio': 
    var = np.log10(np.load(source+'/data_image/ratio_13CO32_13CO21.npy').reshape(-1))
elif input == 'Nco': 
    var = (np.load(source+'/radex_model/Nco_'+model+'_rmcor_'+source+'_los200_median_interp.npy')
            + np.log10(fits.open(source+'/data_image/'+source+'_CO21_ew_broad_nyq.fits')[0].data / 15)).reshape(-1)
elif input == 'Ico': 
    var = np.log10(np.load(source+'/data_image/NGC3351_CO21_mom0.npy').reshape(-1))
    #var[var == 0] = np.nan
elif input == 'nH2': 
    var = np.load(source+'/radex_model/nH2_'+model+'_rmcor_'+source+'_los200_median_interp.npy').reshape(-1)
elif input == 'sqrtn/T':
    var = np.load(source+'/radex_model/sqrt_n_over_T_'+model+'_'+source+'_median_los200.npy').reshape(-1)

alpha = data_alpha_peak * mask 
alpha[mask == 0] = np.nan
alpha_med = data_alpha_50 * mask 
alpha_med[mask == 0] = np.nan
alpha_pos = data_alpha_84 * mask 
alpha_pos[mask == 0] = np.nan
alpha_neg = data_alpha_16 * mask 
alpha_neg[mask == 0] = np.nan

err_y_up = alpha_pos - alpha_med
err_y_low = alpha_med - alpha_neg

if regional:
    mask_2 = mask_whole * np.load(source+'/mask_nucleus.npy')
    mask_2 = mask_2.reshape(-1)
    alpha_2 = data_alpha_peak * mask_2 
    alpha_2[mask_2 == 0] = np.nan
    alpha_med_2 = data_alpha_50 * mask_2 
    alpha_med_2[mask_2 == 0] = np.nan
    alpha_pos_2 = data_alpha_84 * mask_2 
    alpha_pos_2[mask_2 == 0] = np.nan
    alpha_neg_2 = data_alpha_16 * mask_2 
    alpha_neg_2[mask_2 == 0] = np.nan
    err_y_up_2 = alpha_pos_2 - alpha_med_2
    err_y_low_2 = alpha_med_2 - alpha_neg_2

    mask_3 = mask_whole * np.load(source+'/mask_ring.npy')
    mask_3 = mask_3.reshape(-1)
    alpha_3 = data_alpha_peak * mask_3 
    alpha_3[mask_3 == 0] = np.nan
    alpha_med_3 = data_alpha_50 * mask_3 
    alpha_med_3[mask_3 == 0] = np.nan
    alpha_pos_3 = data_alpha_84 * mask_3 
    alpha_pos_3[mask_3 == 0] = np.nan
    alpha_neg_3 = data_alpha_16 * mask_3 
    alpha_neg_3[mask_3 == 0] = np.nan

fig, ax = plt.subplots()
plt.tick_params(axis="x", labelsize=12)
plt.tick_params(axis="y", labelsize=12)

if regional:    
    if compact == False:
        plt.errorbar(var, alpha_med, yerr=[err_y_low,err_y_up], linestyle='', marker='.', c='k', fmt='o', mfc='white', ecolor='0.8', elinewidth=0.5)
        ax.plot(var, alpha, c='darkgreen', marker='_', linestyle='')

        mask_solid = (np.load(source+'/data_image/'+source+'_CO21_mom0.npy') > 50).reshape(-1)
        solid = alpha_med * mask_solid
        solid[mask_solid==0] = np.nan

        ax.plot(var, solid, c='darkblue', marker='.', linestyle='', zorder=99, label='Arms') 
    else:
        ax.plot(var, alpha_med, linestyle='', marker='.', c='darkblue', label='Outer Arms')
        ax.plot(var, alpha, c='darkgreen', marker='_', linestyle='')

    ax.plot(var, alpha_med_3, linestyle='', marker='.', c='darkorange', label='Inner Arms')
    ax.plot(var, alpha_3, c='darkgreen', marker='_', linestyle='')

    ax.plot(var, alpha_med_2, linestyle='', marker='.', c='darkred', label='Nucleus')
    ax.plot(var, alpha_2, c='darkgreen', marker='_', linestyle='')

    #ax.axhline(0.65, c='k', linestyle='--')
    plt.legend(fontsize=12, loc='lower right')  #
else:
    if input == 'tau':
        plt.scatter(var, alpha_med, c=Nco, cmap='inferno')
        plt.colorbar()
    else:
        plt.errorbar(var, alpha_med, yerr=[err_y_low,err_y_up], linestyle='', marker='.', c='k', ecolor='0.8', elinewidth=0.5)
        ax.plot(var, alpha, c='darkgreen', marker='_', linestyle='')

    ax.axhline(0.65, c='darkred', linestyle='--')

if input == 'tau':
    ax.set_xlabel(r'$\log\ \tau_{\rm CO(1-0)}$', fontsize=14)  
elif input == 'radius':
    ax.set_xlabel('Galactocentric Radius (arcsec)', fontsize=14) 
    secax = ax.secondary_xaxis('top', functions=(as2kpc, kpc2as))
    secax.tick_params(axis="x", labelsize=12)
    secax.set_xlabel('Galactocentric Radius (kpc)', fontsize=14)
elif input == 'Tk': 
    ax.set_xlabel(r'$\log\ T_k$ (K)', fontsize=14)  
elif input == 'Nco': 
    ax.set_xlabel(r'$\log\ N_{CO}$  $(cm^{-2})$', fontsize=14)
elif input == 'nH2':
    ax.set_xlabel(r'$\log\ n_{H_2}$  $(cm^{-3})$', fontsize=14)
else:
    ax.set_xlabel(r'$\log(\sqrt{n_{H_2}}/T_k)$', fontsize=14)

if source=='NGC4321':
    plt.errorbar(-0.5, -0.9, yerr=0.3, marker='.', c='k', elinewidth=1)
else:
    plt.errorbar(0.5, -1., yerr=0.3, marker='.', c='k', elinewidth=1)

ax.set_ylabel(r'$\log\ \alpha_{CO}$ ($M_\odot\ (K\ km\ s^{-1}\ pc^2)^{-1}$)', fontsize=14)
plt.tight_layout()
plt.show()

