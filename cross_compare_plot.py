import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats

plot_type = 'bin_2d'  #'scatter'  #'run_med'  #
colorcode = True  #False  #
compare_alpha = False  #'R21'  #'Ico10'  #'Tpeak10'  #

def as2kpc(x, dist_Mpc):
    factor = dist_Mpc * 10**6 / 206265 / 1000
    return x * factor

def plot_medians(data_x, data_y, color, num_bins=10, label=None):
    bins = np.linspace(np.nanmin(data_x), np.nanmax(data_x), num_bins)
    delta = bins[1] - bins[0]
    argsort = np.argsort(data_x)
    idx  = np.digitize(data_x[argsort], bins)
    data_y = data_y[argsort]

    running_median = [np.nanmedian(data_y[idx==k]) for k in range(num_bins)]
    plt.plot(bins-delta/2, running_median, color=color, ls='-', lw=2, marker='o', ms=5, label=label)

    running_prc25 = [np.nanpercentile(data_y[idx==k], 25) for k in range(num_bins)]
    running_prc75 = [np.nanpercentile(data_y[idx==k], 75) for k in range(num_bins)]
    plt.fill_between(bins-delta/2, running_prc25, running_prc75, color=color, alpha=0.2)

    print(running_prc25, running_prc75)
    return

def alpha_Teng(logtau, logT):
    alpha = 0.78 * logtau - 0.18 * logT - 0.84
    return alpha

labels = np.array((r'$\log\ T_{k}$ (K)', r'$\log\ \Delta v_{CO(2-1)}$ (km/s)', r'$\log\ I_{CO(1-0)}$ (K km s$^{-1}$)', r'$\log\ \alpha_{CO}$', r'$\log\ \tau_{CO(2-1)}$', 
        r'$\log\ n_{H_2}$ (cm$^{-3}$)', r'$\log\ n_{H_2}^{(3\ lines)}$', r'$\log(\sqrt{n_{H_2}} / T_k)$', r'$X_{12/13}$', r'$\log$ CO/$^{13}$CO 2-1', r'modeled CO 3-2/2-1', 
        r'$\log\ R_{21}$', r'$\log \left[ N_{CO}^{(6\ lines)}\cdot\frac{15\ km\ s^{-1}}{\Delta v}\right]$', r'$\log \left[ N_{CO} \cdot\frac{15\ km\ s^{-1}}{\Delta v}\right]$')) 

data_x = [np.full((75,75), np.nan), np.full((105,105), np.nan), np.full((125,125), np.nan)]
data_y = [np.full((75,75), np.nan), np.full((105,105), np.nan), np.full((125,125), np.nan)]
color = [np.full((75,75), np.nan), np.full((105,105), np.nan), np.full((125,125), np.nan)]

distance = np.array((9.96, 11.32, 15.21))  #Mpc
resolution = np.array((101.4, 109.21, 123.15))  #pc
inclination = np.array((45.1, 57.3, 38.5)) * np.pi/180
R_eff = np.array((3., 3.6, 5.5))  #kpc

# Load data
for count, source in enumerate(['NGC3351','NGC3627', 'NGC4321']):  #  
    if count == 0:
        if colorcode:
            color[count] = np.log10(np.load(source+'/data_image/ratio_CO21_13CO21_broad.npy')) 

        mask = np.load(source+'/mask_whole_recovered.npy') * np.load(source+'/mask_armscent.npy') * np.load(source+'/mask_rmcor_comb_lowchi2.npy') 
            #* np.isfinite(np.log10(np.load(source+'/data_image/ratio_CO21_13CO21_broad.npy')))        
        data_x[count] = np.load(source+'/radex_model/tau_6d_coarse2_'+source+'_co21_median_los200.npy') * mask  
        data_y[count] = np.load(source+'/radex_model/Xco_6d_coarse_ewsame_median_los100.npy') * mask  
        data_x[count][mask==0] = np.nan
        data_x[count][~np.isfinite(data_x[count])] = np.nan
        data_y[count][mask==0] = np.nan 

    else:
        if colorcode:
            color[count] = np.log10(np.load(source+'/data_image/ratio_CO21_13CO21_broad.npy')) 

        mask = np.load(source+'/mask_13co21_3sig.npy') * np.load(source+'/mask_recovered_0.3.npy') * (np.load(source+'/data_image/'+source+'_CO21_mom0.npy') > 50) 
        data_x[count] = np.load(source+'/radex_model/tau_6d_coarse2_'+source+'_co21_median_los200.npy') * mask  
        data_y[count] = np.load(source+'/radex_model/Xco_6d_coarse2_'+source+'_ewsame_median_los200.npy') * mask  
        data_x[count][mask==0] = np.nan
        data_y[count][mask==0] = np.nan

if compare_alpha:    
    Gong_alphaCO = np.log10(np.load('Gong20_simulation/alphaCO_full_2pc.npy'))
    Gong_alphaCO_2 = np.log10(np.load('Gong20_simulation/alphaCO_full_128pc.npy'))
    Gong_data = np.log10(np.load('Gong20_simulation/'+compare_alpha+'_full_2pc.npy'))
    Gong_data_2 = np.log10(np.load('Gong20_simulation/'+compare_alpha+'_full_128pc.npy'))

    if compare_alpha == 'Ico10':
        x_range_G20 = np.arange(-0.3, 2.4, 0.01)  
        x_range_N12 = np.arange(1.3, 3.5, 0.01)    
        alpha_Gong = np.log10(6.1 * (10**x_range_G20)**(-0.54+0.19*2) * 100**-0.25 * 10/4.5)  #assume Z=1 and consistent beam size of 100 pc
        alpha_Narayanan = np.log10(1.36 * 10.7 * (10**x_range_N12)**-0.32)  #assume Z=1

        Hu_Ico = np.array((-3., -2.5, -2., -1.5, -1., -0.5, 0., 0.5, 1., 1.36))
        Hu_alpha = np.log10(10**np.array((21.913, 21.641, 21.378, 21.098, 20.817, 20.588, 20.401, 20.308, 20.189, 20.104)) / 4.5e19)

    elif compare_alpha == 'R21':
        x_range_G20 = np.arange(-1., 0.2, 0.01)  
        alpha_Gong = np.log10(0.93 * (10**x_range_G20/0.6)**-0.87 * 100**0.081 * 10/4.5)  #assume Z=1

    elif compare_alpha == 'Tpeak10':
        x_range_G20 = np.arange(-1.35, 1.1, 0.01)  #
        alpha_Gong = np.log10(1.8 * (10**x_range_G20)**(-0.64+0.24*2) * 100**-0.083 * 10/4.5)  #assume Z=1 and consistent beam size of 100 pc
    

fig, ax = plt.subplots()
plt.tick_params(axis="x", labelsize=14)
plt.tick_params(axis="y", labelsize=14, labelleft=True)  #

if plot_type == 'run_med':
    total_bins = 10
    for count, (source, color) in enumerate(zip(['NGC3351', 'NGC3627', 'NGC4321'], ['r','b','g'])):  # 
        data_x_flat = data_x[count].reshape(-1)
        data_y_flat = data_y[count].reshape(-1)
        if compare_alpha == 'Ico10':
            plot_medians(data_x_flat, data_y_flat, color, total_bins, None) 
        else:
            plot_medians(data_x_flat, data_y_flat, color, total_bins, source) 
        

elif plot_type == 'scatter':
    if colorcode:
        plt.scatter(data_x[0], data_y[0], c=color[0], s=5, cmap='inferno')  
        plt.scatter(data_x[1], data_y[1], c=color[1], s=5, cmap='inferno')         
        plt.scatter(data_x[2], data_y[2], c=color[2], s=5, cmap='inferno')     
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        cb.ax.set_title(labels[0], fontsize=14)
    else: 
        data_x[0][data_x[0] < -1.5] = np.nan
        plt.scatter(data_x[0], data_y[0], c='darkred', s=5, label='NGC3351') 
        plt.scatter(data_x[1], data_y[1], c='darkblue', s=5, label='NGC3627')  #, alpha=0.8
        plt.scatter(data_x[2], data_y[2], c='darkgreen', s=5, label='NGC4321')  #, alpha=0.5

elif plot_type == 'bin_2d':
    X = np.concatenate((data_x[0].reshape(-1), data_x[1].reshape(-1), data_x[2].reshape(-1)))
    Y = np.concatenate((data_y[0].reshape(-1), data_y[1].reshape(-1), data_y[2].reshape(-1)))
    COLOR = np.concatenate((color[0].reshape(-1), color[1].reshape(-1), color[2].reshape(-1)))
    
    idx_finite = np.isfinite(X) * np.isfinite(Y) * np.isfinite(COLOR)
    X = X[idx_finite]
    Y = Y[idx_finite]
    COLOR = COLOR[idx_finite]

    total_bins = 30
    binx = np.linspace(-1.5, 1.8, 34)
    biny = np.linspace(-1.8, 1.1, 30) #np.linspace(np.nanmin(Y), np.nanmax(Y), total_bins)
    ret = stats.binned_statistic_2d(X, Y, COLOR, 'median', bins=[binx, biny], expand_binnumbers=True)
    color_binned = ret.statistic[ret.binnumber[0] - 1, ret.binnumber[1] - 1]
    
    plt.scatter(X, Y, c=color_binned, s=5, cmap='inferno', vmin=0.65, vmax=1.82)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    cb.ax.set_title(labels[9], fontsize=14)
    cb.ax.plot([0,1], [0.78]*2, 'w', lw=1)

## Optional add ons for the plot
# plt.axhline(0., c='k', linestyle='--')
# plt.axvline(np.log10(5), c='k', linestyle=':')
plt.plot(np.log10(0.82), np.log10(0.08), marker='+', c='k', mew=2, ms=12)

## alpha vs ratio21
# plt.plot(np.arange(0.6, 1.8, 0.1), (np.arange(0.6, 1.8, 0.1) * -0.40 + 0.23), 'k--')
# plt.plot(np.log10(6), np.log10(4.35), marker='*', c='tab:blue', mew=2, ms=10)
# plt.annotate('MW disk', weight='bold', fontsize=12, xy=(0.69, 0.48), xycoords='data', color='tab:blue') 

## alpha vs ew21
# plt.plot(np.arange(0.85, 1.95, 0.1), (np.arange(0.85, 1.95, 0.1) * -0.63 + 0.61), 'k--') 
# plt.plot(np.log10(5), np.log10(4.35), marker='*', c='tab:blue', mew=2, ms=10)
# plt.annotate('MW disk', weight='bold', fontsize=12, xy=(0.68, 0.48), xycoords='data', color='tab:blue')

if compare_alpha:
    plt.plot(x_range_G20, alpha_Gong, 'k--', lw=1.5)
    plot_medians(Gong_data, Gong_alphaCO, color='0.5', label='Gong+20 (2-pc)')
    plot_medians(Gong_data_2, Gong_alphaCO_2, 'tab:brown', label='Gong+20 (128-pc)') 

    if compare_alpha == 'Ico10':
        plt.plot(Hu_Ico, Hu_alpha, color='tab:olive', ls='-', lw=2, marker='o', ms=5, label='Hu+22 (125-pc)')
        plt.plot(x_range_N12, alpha_Narayanan, 'k:', lw=2, label='Narayanan+12')
        plt.xlim(-0.7,3.2)
        plt.xlabel(labels[2], fontsize=16)

if (colorcode==False or plot_type=='run_med'):
    plt.legend(fontsize=14)  #, loc='lower left'

plt.xlabel(labels[4], fontsize=16)  #'Galactocentric Radius (kpc)'
plt.ylabel('', fontsize=16) 

plt.savefig('bin2d_alpha_vs_tau21_whole3351_color_ratio21_ewsame_plus.pdf', bbox_inches='tight', pad_inches=0.02)
plt.show()
