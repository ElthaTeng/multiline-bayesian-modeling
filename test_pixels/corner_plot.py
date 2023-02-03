import numpy as np
import matplotlib.pyplot as plt
import corner
import time
from astropy.io import fits

'''
This script generates a corner plot showing 1D and 2D likelihood distributions for
all six parameters (dimensions) by loading in the chi^2/probability arrays returned 
from 'bestfit_onepix.py' or 'bestfit_onepix_cov.py'. Note that 'bestfit_onepix.py'
only saves the chi^2 array, so the probability should be computed in this script.

'''
source = 'NGC4321'
start_time = time.time()
print('Constructing inputs for corner plot...')

stack = False  # Set this
model = '6d_coarse2_rmcor'
sou_model = 'radex_model/'
max1d = np.array((16.0,1.0,3.6,30,14,0.05))  # Optional
bestfit = np.array((18.2,2.3,3.4,70,4,0))  # Optional

if stack:
    region = 'arms'  # Set this
    chi2 = np.load(sou_model+'chi2_'+model+'_'+region+'.npy').reshape(-1)
else:
    idx_x = 61  # Set this
    idx_y = 62  # Set this  
    chi2 = np.load(sou_model+'chi2_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'.npy').reshape(-1)

prob = np.exp(-0.5*chi2)
prob[np.isnan(prob)] = 0

N_co = np.arange(15., 20.1, 0.2)
T_k = np.arange(1.,2.8,0.1)
n_h2 = np.arange(2., 5.1, 0.2)
X_12to13 = np.arange(10, 201, 10)
X_13to18 = np.arange(2, 21, 1)
phi = np.arange(-1.3, 0.1, 0.1)

size_N = N_co.shape[0]
size_T = T_k.shape[0]
size_n = n_h2.shape[0]
size_x12to13 = X_12to13.shape[0]
size_x13to18 = X_13to18.shape[0]
size_phi = phi.shape[0]

N_co = N_co.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
T_k = T_k.reshape(1,size_T,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
n_h2 = n_h2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
X_12to13 = X_12to13.reshape(1,1,1,size_x12to13,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
X_13to18 = X_13to18.reshape(1,1,1,1,size_x13to18,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
phi = phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))

N_co = N_co.reshape(-1)
T_k = T_k.reshape(-1)
n_h2 = n_h2.reshape(-1)
X_12to13 = X_12to13.reshape(-1)
X_13to18 = X_13to18.reshape(-1)
phi = phi.reshape(-1)

# Set constraints to exclude unreasonable parameter sets
los_max = 200.
x_co = 3 * 10**(-4)
map_ew = fits.open('data_image/'+source+'_CO21_ew_broad_nyq.fits')[0].data
map_fwhm = map_ew * 2.35  # Conversion of 1 sigma to FWHM assuming Gaussian
if stack:
    los_length = (10**N_co / 15. * np.load('data_cube/fitting_'+region+'.npy')[1,2]) / (np.sqrt(10**phi) * 10**n_h2 * x_co)
else:
    los_length = (10**N_co / 15. * map_fwhm[idx_y,idx_x]) / (np.sqrt(10**phi) * 10**n_h2 * x_co)  

mask = los_length < los_max * (3.086 * 10**18)  
print(np.sum(mask),'parameter sets have line-of-sight length smaller than',los_max,'pc')

if stack:
    np.save(sou_model+'mask_'+model+'_'+region+'_los200.npy', mask)
    print('Mask of path length <',los_max,'for',region,'region saved.')

chi2_masked = chi2 * mask
chi2_masked[mask == 0] = np.nan
idx_min = np.unravel_index(np.nanargmin(chi2_masked), (size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
print('Minumum chi^2 =', np.nanmin(chi2_masked), 'at', idx_min)

par_min = np.asarray(idx_min)
Nco = np.round(0.2*par_min[0] + 15., 1)
Tk = np.round(0.1*par_min[1] + 1., 1)
nH2 = np.round(0.2*par_min[2] + 2., 1)
X12to13 = np.round(10*par_min[3] + 10., 1)
X13to18 = np.round(1*par_min[4] + 2., 1)
Phi = np.round(0.1*par_min[5] - 1.3, 1)
print('i.e. (Nco, Tk, nH2, X(12/13), X(13/18), Phi) =', Nco, Tk, nH2, X12to13, X13to18, Phi)

prob = prob * mask
print(np.isnan(prob).sum())

# Corner plot
ndim = 6
input = np.stack((N_co, T_k, n_h2, X_12to13, X_13to18, phi), axis=-1)

corner_time = time.time()
print(corner_time-start_time, 'sec elapsed. Generating corner plot')
figure = corner.corner(input, bins=[size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi], plot_datapoints=False, quantiles=[0.5],
                       labels=[r"$\log \left( N_{CO}\cdot\frac{15\ km\ s^{-1}}{\Delta v}\right)$", r"$\log(T_k)$", r"$\log(n_{H_2})$", r"$X_{12/13}$", r"$X_{13/18}$", r"$\log(\Phi_{bf})$"], range=[(14.9,20.1),(0.95,2.75),(1.9,5.1),(5,205),(1.5,20.5),(-1.35,0.05)],
                       label_kwargs={"fontsize": 16}, show_titles=True, title_fmt=None, title_kwargs={"fontsize": 18}, weights=prob)
axes = np.array(figure.axes).reshape((ndim, ndim))
for i in range(ndim):
    ax = axes[i, i]
    #ax.axvline(max1d[i], color="k", linestyle='dashed', linewidth=2)
    ax.axvline(bestfit[i], color="k", linestyle='dotted', linewidth=1.5)
    ax.grid(b=True, axis='x', color='lightgrey', linestyle='dotted', linewidth=1)
    ax.tick_params(axis="x", labelsize=14)
for yi in range(ndim):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.grid(b=True, color='lightgrey', linestyle='dotted', linewidth=1)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
if stack:
    plt.savefig(sou_model+'corner_'+model+'_'+region+'_los200_vline_grid.pdf')
else:
    plt.savefig(sou_model+'corner_los200_'+model+'_'+str(idx_x)+'_'+str(idx_y)+'_cal10-20_bf.pdf')

end_time = time.time()
print(end_time-corner_time, 'sec elapsed. Corner plot saved.')
print('Total time elapsed =', end_time-start_time, 'sec')

plt.show(figure)


