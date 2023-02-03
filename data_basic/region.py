import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch
from astropy.io import fits
from astropy.wcs import WCS
import copy

source = 'NGC3627'

fits_map = fits.open(source+'/data_image/'+source+'_CO21_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0].header)
mom0 = fits_map[0].data
mask = np.load(source+'/mask_recovered_0.3.npy') * np.load(source+'/mask_13co21_3sig.npy') * (mom0 > 50)

reg_nucl = (np.load(source+'/mask_nucleus.npy') * mask).astype('float')
reg_ring = (np.load(source+'/mask_ring.npy') * mask).astype('float')
reg_inflow = (np.load(source+'/mask_inflow.npy') * mask).astype('float')
radius = np.load(source+'/'+source+'_radius_arcsec.npy')

reg_nucl[reg_nucl==0] = np.nan
reg_ring[reg_ring==0] = np.nan
reg_inflow[reg_inflow==0] = np.nan

cmap_nucl = copy.copy(plt.cm.get_cmap('cool')) 
cmap_nucl.set_bad(alpha=0) 

cmap_ring = copy.copy(plt.cm.get_cmap('viridis')) 
cmap_ring.set_bad(alpha=0) 

cmap_inflow = copy.copy(plt.cm.get_cmap('cool_r')) 
cmap_inflow.set_bad(alpha=0) 


fig = plt.figure()
ax = fig.add_subplot(111, projection=wcs) #
ra = ax.coords[0]
ra.set_major_formatter('hh:mm:ss.s')

plt.imshow(reg_nucl, origin='lower', cmap='cool', vmin=0, vmax=1)
plt.imshow(reg_ring, origin='lower', cmap=cmap_ring, vmin=0, vmax=1)
plt.imshow(reg_inflow, origin='lower', cmap=cmap_inflow, vmin=0, vmax=1)

plt.tick_params(axis="y", labelleft=True, labelsize=14)
plt.tick_params(axis="x", labelbottom=True, labelsize=14)

ring = Patch(facecolor='gold')
nucleus = Patch(facecolor='magenta')
inflow = Patch(facecolor='cyan')
null = Patch(facecolor='w')

lprop = {'weight':'bold', 'size':'large'}
plt.legend(handles=[nucleus, ring, inflow],
          labels=['Nucleus', 'Inner arms', 'Outer arms'],
          ncol=1, handletextpad=0.5, handlelength=1.0, columnspacing=0.5, prop=lprop,
          loc='lower right', fontsize=14)

if source == 'NGC4321':
    plt.contour(mom0, origin='lower', levels=(50,100,200,300,500,700), extent=[-0.5,124.5,-0.5,124.5], colors='k', linewidths=1)
    cont_radius = plt.contour(radius,origin='lower',levels=(3,10,20), extent=[-0.5,124.5,-0.5,124.5], colors='gray', linewidths=2)
    plt.xlim(30,90)   
    plt.ylim(30,90)
elif source == 'NGC3627':
    plt.contour(mom0, origin='lower', levels=(50,100,200,300,500,700,900), extent=[-0.5,104.5,-0.5,104.5], colors='k', linewidths=1)
    cont_radius = plt.contour(radius,origin='lower',levels=(3,10,20), extent=[-0.5,104.5,-0.5,104.5], colors='gray', linewidths=2)
    plt.xlim(20,80)   
    plt.ylim(20,80)  

#manual_locations = [(44, 41), (51, 42)]
plt.gca().clabel(cont_radius, inline=1, fontsize=10, fmt='%1.0f') #, manual=manual_locations

plt.title(source, fontsize=18)
plt.xlabel('R.A. (J2000)', fontsize=16) #
plt.ylabel('Decl. (J2000)', fontsize=16) #
plt.savefig('region_def_'+source+'.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

