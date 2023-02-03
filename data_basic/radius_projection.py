import warnings
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import Angle

def radius_arcsec(shape, w, ra, dec, pa, incl,
                  incl_correction=False, cosINCL_limit=0.5):
    # All inputs assumed as Angle
    if incl_correction and (np.isnan(pa.rad + incl.rad)):
        pa = Angle(0 * u.rad)
        incl = Angle(0 * u.rad)
        # Not written to the header
        msg = '\n::z0mgs:: PA or INCL is NaN in ' + \
            'radius calculation \n' + \
            '::z0mgs:: Setting both to zero.'
        # Warning message ends
        warnings.warn(msg, UserWarning)
        # Warning ends
    cosPA, sinPA = np.cos(pa.rad), np.sin(pa.rad)
    cosINCL = np.cos(incl.rad)
    if incl_correction and (cosINCL < cosINCL_limit):
        cosINCL = cosINCL_limit
        # Not written to the header
        msg = '\n::z0mgs:: Large inclination encountered in ' + \
            'radius calculation \n' + \
            '::z0mgs:: Input inclination: ' + str(incl.deg) + \
            ' degrees. \n' + \
            '::z0mgs:: cos(incl) is set to ' + str(cosINCL_limit)
        # Warning message ends
        warnings.warn(msg, UserWarning)
        # Warning ends
    xcm, ycm = ra.rad, dec.rad

    dp_coords = np.zeros(list(shape) + [2])
    # Original coordinate is (y, x)
    # :1 --> x, RA --> the one needed to be divided by cos(incl)
    # :0 --> y, Dec
    dp_coords[:, :, 0], dp_coords[:, :, 1] = \
        np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # Now, value inside dp_coords is (x, y)
    # :0 --> x, RA --> the one needed to be divided by cos(incl)
    # :1 --> y, Dec
    for i in range(shape[0]):
        dp_coords[i] = Angle(w.wcs_pix2world(dp_coords[i], 1) * u.deg).rad
    dp_coords[:, :, 0] = 0.5 * (dp_coords[:, :, 0] - xcm) * \
        (np.cos(dp_coords[:, :, 1]) + np.cos(ycm))
    dp_coords[:, :, 1] -= ycm
    # Now, dp_coords is (dx, dy) in the original coordinate
    # cosPA*dy-sinPA*dx is new y
    # cosPA*dx+sinPA*dy is new x
    radius = np.sqrt((cosPA * dp_coords[:, :, 1] +
                      sinPA * dp_coords[:, :, 0])**2 +
                     ((cosPA * dp_coords[:, :, 0] -
                       sinPA * dp_coords[:, :, 1]) / cosINCL)**2)
    radius = Angle(radius * u.rad).arcsec
    return radius

source = 'NGC4321'    
fits_map = fits.open(source+'/data_image/'+source+'_CO21_mom0_broad_nyq.fits')
wcs = WCS(fits_map[0])
dshape = fits_map[0].data.shape

if source == 'NGC3627':
    angles = Angle(['170.06252d','12.9915d','173.1d','57.3d'])  #ra, dec, pa, incl
elif source == 'NGC4321':
    angles = Angle(['185.72886d','15.8223d','156.2d','38.5d']) 

radii_map = np.full(dshape, np.nan)
radii_map[:-1, :-1] = radius_arcsec(dshape, wcs, angles[0], angles[1], angles[2], angles[3])[1:, 1:]
np.save(source+'/'+source+'_radius_arcsec.npy',radii_map)

nucleus = radii_map < 3
ring = (radii_map > 3) * (radii_map < 10)
inflow = (radii_map > 10) * (fits_map[0].data > 50)

np.save(source+'/mask_nucleus.npy', nucleus)
np.save(source+'/mask_ring.npy', ring)
np.save(source+'/mask_inflow.npy', inflow)

plt.imshow(inflow, origin='lower')
plt.colorbar()
plt.contour(fits_map[0].data, origin='lower', levels=(50,100,200,300,500,700), extent=[-0.5,dshape[1]-0.5,-0.5,dshape[0]-0.5], colors='dimgray', linewidths=1)  # 
plt.show()
