import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

source = 'NGC3627'
par = 'dv'  # ratio, mom0, 1dmax, median_interp, alpha
region = 'whole'  # nucleus, ring, inflow, whole
mom0_co21 = np.load(source+'/data_image/'+source+'_CO21_mom0.npy')
mask_whole = np.load(source+'/mask_recovered_0.3.npy') * np.load(source+'/mask_13co21_3sig.npy') * (mom0_co21 > 50)
shape = mask_whole.shape[0]
mask_c18o = np.load(source+'/mask_c18o21_3sig.npy')

if par == 'ratio':
    maps = np.full((7,shape,shape), np.nan)
    line_up = np.array(('CO21', '13CO32', 'C18O32', 'CO21', 'CO21', '13CO21', '13CO32'))
    line_low = np.array(('CO10', '13CO21', 'C18O21', '13CO21', 'C18O21', 'C18O21', 'C18O32'))
    for i in range(7):    
        map_up = np.load(source+'/data_image/'+source+'_'+line_up[i]+'_mom0.npy')
        map_low = np.load(source+'/data_image/'+source+'_'+line_low[i]+'_mom0.npy')

        map = map_up/map_low
        map[map<=0] = np.nan
        maps[i] = map
        
elif par == 'mom0':
    maps = np.full((6,shape,shape), np.nan)
    maps[0] = np.load(source+'/data_image/'+source+'_CO10_mom0.npy')
    maps[1] = np.load(source+'/data_image/'+source+'_CO21_mom0.npy')
    maps[2] = np.load(source+'/data_image/'+source+'_13CO21_mom0.npy')
    maps[3] = np.load(source+'/data_image/'+source+'_13CO32_mom0.npy')
    maps[4] = np.load(source+'/data_image/'+source+'_C18O21_mom0.npy')
    maps[5] = np.load(source+'/data_image/'+source+'_C18O32_mom0.npy')

        
elif par == '1dmax':
    maps = np.full((6,shape,shape), np.nan)
    maps[0] = np.load(source+'/radex_model/Nco_6d_coarse2_rmcor_'+source+'_los200_1dmax.npy')
    maps[1] = np.load(source+'/radex_model/Tk_6d_coarse2_rmcor_'+source+'_los200_1dmax.npy')
    maps[2] = np.load(source+'/radex_model/nH2_6d_coarse2_rmcor_'+source+'_los200_1dmax.npy')
    maps[3] = np.load(source+'/radex_model/X12to13_6d_coarse2_rmcor_'+source+'_los200_1dmax.npy')
    maps[4] = np.load(source+'/radex_model/X13to18_6d_coarse2_rmcor_'+source+'_los200_1dmax.npy') 
    maps[5] = np.load(source+'/radex_model/phi_6d_coarse2_rmcor_'+source+'_los200_1dmax.npy')

    maps[4] = maps[4] * mask_c18o
    maps[4][mask_c18o==0] = np.nan
    
elif par == 'median_interp':
    maps = np.full((6,shape,shape), np.nan)
    maps[0] = np.load(source+'/radex_model/Nco_6d_coarse2_rmcor_'+source+'_los200_median_interp.npy')
    maps[1] = np.load(source+'/radex_model/Tk_6d_coarse2_rmcor_'+source+'_los200_median_interp.npy')
    maps[2] = np.load(source+'/radex_model/nH2_6d_coarse2_rmcor_'+source+'_los200_median_interp.npy')
    maps[3] = np.load(source+'/radex_model/X12to13_6d_coarse2_rmcor_'+source+'_los200_median_interp.npy')
    maps[4] = np.load(source+'/radex_model/X13to18_6d_coarse2_rmcor_'+source+'_los200_median_interp.npy') 
    maps[5] = np.load(source+'/radex_model/phi_6d_coarse2_rmcor_'+source+'_los200_median_interp.npy')

    maps[4] = maps[4] * mask_c18o
    maps[4][mask_c18o==0] = np.nan

else:
    maps = np.full((1,shape,shape), np.nan)
    maps[0] = fits.open(source+'/data_image/'+source+'_CO21_ew_broad_nyq.fits')[0].data #np.load(source+'/radex_model/Xco_6d_coarse2_alpha_median_los200.npy')
    
if region == 'nucleus':
    mask = np.load(source+'/mask_nucleus.npy') * mask_whole * (mom0_co21 > 50)
elif region == 'ring':
    mask = np.load(source+'/mask_ring.npy') * mask_whole * (mom0_co21 > 50)
elif region == 'inflow':
    mask = np.load(source+'/mask_inflow.npy') * mask_whole * (mom0_co21 > 50)
else:
    mask = mask_whole 
  
maps = maps * mask
maps[:,mask==0] = np.nan  #
maps[np.isinf(maps)] = np.nan 

if par == 'mom0':
    index_ratio = np.array(((1,0), (3,2), (5,4), (1,2), (1,4), (2,4), (3,5)))
    ratios = np.full((7,), np.nan)
    
    for i in range(7):
        mask_both3sig = (maps[index_ratio[i,0]] > 0) * (maps[index_ratio[i,1]] > 0)
        mom0_up = np.nansum(maps[index_ratio[i,0]] * mask_both3sig)
        mom0_low = np.nansum(maps[index_ratio[i,1]] * mask_both3sig)
        ratios[i] = mom0_up / mom0_low
    
    print(ratios)

else:
    print(np.nanmedian(maps, axis=(1,2)))
    print(np.nanmean(maps, axis=(1,2)))
    print('+/-', np.nanstd(maps, axis=(1,2)))

plt.imshow(maps[0], origin='lower', cmap='jet')
plt.colorbar()
plt.show()