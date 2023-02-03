import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

source = 'NGC4321'
line = np.array(('CO10','CO21','13CO21','13CO32','C18O21','C18O32'))
title = np.array(('(a) CO 1-0','(b) CO 2-1',r'(c) $\rm {}^{13}$CO 2-1',r'(d) $\rm {}^{13}$CO 3-2',r'(e) $\rm {C}^{18}$O 2-1',r'(f) $\rm {C}^{18}$O 3-2'))

fits_map = fits.open('data_image/'+source+'_'+line[2]+'_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0].header)
vmax = np.array((900, 50, 14))

fig = plt.figure(figsize=(18,10))

for i in range(6):

    mom0 = np.load('data_image/'+source+'_'+line[i]+'_mom0.npy')
    ax = fig.add_subplot(2,3,i+1, projection=wcs)
    ra = ax.coords[0]
    ra.set_major_formatter('hh:mm:ss.s')
    
    zeros = np.ma.masked_where(~(mom0 == 0), mom0)
    plt.imshow(mom0, cmap='hot', origin='lower', vmax=vmax[i//2])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    plt.imshow(zeros, origin='lower', cmap='Pastel2_r')
    
    if i == 2:
        beam = plt.Circle((84, 36), 1, color='b')
        ax.add_patch(beam)
        plt.plot([32, 48], [34, 34], 'b-', lw=3)
        plt.annotate('1 kpc', weight='bold', fontsize=14, xy=(35, 36), xycoords='data', color='b')
        
    plt.title(title[i], fontsize=18)
    plt.xlim(30,90)   
    plt.ylim(30,90)  
    plt.tick_params(axis="x", labelsize=14)
    plt.tick_params(axis="y", labelsize=14)
    
    if i//3 == 0:
        plt.xlabel(' ')
        plt.tick_params(axis="x", labelbottom=False)
    else:
        plt.xlabel('R.A. (J2000)', fontsize=16)
    if i%3 == 0:
        plt.ylabel('Decl. (J2000)', fontsize=16)
    else:
        plt.ylabel(' ')
        plt.tick_params(axis="y", labelleft=False)
    
plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.1)
plt.savefig('std_plots/NGC4321_mom0_subplots_all3sig.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
