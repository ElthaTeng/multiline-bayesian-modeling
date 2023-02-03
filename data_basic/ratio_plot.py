import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits

source = 'NGC4321'
mom0 = np.load('data_image/'+source+'_CO21_mom0.npy')

line_up = 'CO21'
line_low = 'C18O21'
map_up = np.load('data_image/'+source+'_'+line_up+'_mom0.npy')
map_low = np.load('data_image/'+source+'_'+line_low+'_mom0.npy')

fits_map = fits.open('data_image/'+source+'_CO21_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0].header)

ratio = map_up/map_low
ratio[ratio<=0] = np.nan
np.save('data_image/'+source+'_ratio_'+line_up+'_'+line_low+'.npy', ratio)

mask = np.load('mask_recovered_0.3.npy')
ratio_masked = ratio * mask
ratio_masked[mask==0] = np.nan

fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs)
ra = ax.coords[0]
ra.set_major_formatter('hh:mm:ss.s')
plt.imshow(ratio_masked, cmap='Spectral_r', origin='lower', vmin=20, vmax=200)  #
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14)
plt.contour(mom0,origin='lower',levels=(50,100,200,300,500,700), extent=[-0.5,124.5,-0.5,124.5], colors='k', linewidths=1)
plt.title(r'(e) CO/$\rm {C}^{18}O$ 2-1', fontsize=18) 

plt.xlim(30,90)   
plt.ylim(30,90)  
plt.tick_params(axis="x", labelsize=14)
plt.tick_params(axis="y", labelsize=14)
plt.xlabel('R.A. (J2000)', fontsize=14)
plt.ylabel('Decl. (J2000)', fontsize=14)
plt.savefig('std_plots/line_ratios/'+source+'_'+line_up+'_'+line_low+'_ratio.pdf', bbox_inches='tight', pad_inches=0.02)
plt.show()