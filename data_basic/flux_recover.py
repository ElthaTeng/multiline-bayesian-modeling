import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.io import fits

source = 'NGC4321'
masking = True
thres = 0.3  # Set only when masking == True

def get_mask(image, cutoff):
    mask = (image > (1 - cutoff)) * (image < (1 + cutoff))
    return mask

sou_data = source + '/data_image/'
fits_map = fits.open(sou_data+source+'_CO21_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0].header)

map_12m = fits.open(sou_data+source+'_CO21_12m_mom0_broad_nyq.fits')[0].data
map_comb = fits_map[0].data

ratio = map_12m/map_comb
ratio[ratio<=0] = np.nan

np.save(source+'/flux_recover_12m_comb.npy', ratio)

if masking:
    mask = get_mask(ratio, thres)
    np.save(source+'/mask_recovered_'+str(thres)+'.npy', mask)

fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs)
ra = ax.coords[0]
ra.set_major_formatter('hh:mm:ss.s')

if masking:
    ratio = ratio*mask
    ratio[mask==0] = np.nan

plt.imshow(ratio, vmin=0.5, vmax=1.5, cmap='Spectral_r', origin='lower')
plt.tick_params(axis="y", labelsize=14, labelleft=True)
plt.tick_params(axis="x", labelsize=14, labelbottom=True)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)

plt.contour(map_comb,origin='lower',levels=(50,100,200,300,500,700), colors='k', linewidths=1)
plt.title(source+' Flux Recover', fontsize=16) 
if source == 'NGC4321':
    plt.xlim(30,90)   
    plt.ylim(30,90)
else:
    plt.xlim(20,80)   
    plt.ylim(20,80)  
plt.xlabel('R.A. (J2000)', fontsize=14)
plt.ylabel('Decl. (J2000)', fontsize=14)

if masking:
    plt.savefig(source+'/std_plots/'+source+'_flux_recover_'+str(thres)+'.pdf', bbox_inches='tight', pad_inches=0.1)
else:
    plt.savefig(source+'/std_plots/'+source+'_flux_recover_12m_comb.pdf', bbox_inches='tight', pad_inches=0.1)

plt.show()
