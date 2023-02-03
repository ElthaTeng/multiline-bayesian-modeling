import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

idx_x = 51
idx_y = 57
source = 'NGC3627'
model = '6d_coarse2'
sou_model = 'radex_model/'
sou_data = 'data_image/'

# Set parameter ranges
samples_Nco = np.arange(15.,20.1,0.2)
samples_Tk = np.arange(1.,2.8,0.1)
samples_nH2 = np.arange(2.,5.1,0.2)
samples_X12to13 = np.arange(10,205,10)
samples_X13to18 = np.arange(2,21,1)
samples_phi = np.arange(-1.3, 0.1, 0.1)

size_N = samples_Nco.shape[0]
size_T = samples_Tk.shape[0]
size_n = samples_nH2.shape[0]
size_x12to13 = samples_X12to13.shape[0]
size_x13to18 = samples_X13to18.shape[0]
size_phi = samples_phi.shape[0]

Nco = samples_Nco.reshape(size_N,1,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
Nco = Nco.reshape(-1)
Tk = samples_Tk.reshape(1,size_T,1,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
Tk = Tk.reshape(-1)
nH2 = samples_nH2.reshape(1,1,size_n,1,1,1)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
nH2 = nH2.reshape(-1)
phi = samples_phi.reshape(1,1,1,1,1,size_phi)*np.ones((size_N,size_T,size_n,size_x12to13,size_x13to18,size_phi))
phi = phi.reshape(-1)

# Set up constraints if using priors
los_max = 200.
x_co = 3 * 10**(-4)
map_ew = fits.open(sou_data+source+'_CO10_ew_broad_nyq.fits')[0].data 
map_fwhm = map_ew * 2.35  # Conversion of 1 sigma to FWHM assuming Gaussian
los_length = (10**Nco / 15. * map_fwhm[idx_y,idx_x]) / (np.sqrt(10**phi) * 10**nH2 * x_co)  
mask = los_length < los_max * (3.086 * 10**18) 
print(np.sum(mask),'parameter sets have line-of-sight length smaller than',los_max,'pc')

# Generate masked sqrt(n)/T grid
grid = nH2 / 2 - Tk  # i.e., log[sqrt(n)/T]
grid[mask == 0] = np.nan
print(np.nanmax(grid), np.nanmin(grid))

# Compute masked chi2 and prob
chi2 = np.load(sou_model+'chi2_'+model+'_rmcor_'+str(idx_x)+'_'+str(idx_y)+'.npy').reshape(-1)
prob = np.exp(-0.5*chi2).reshape(-1)
chi2_masked = chi2 * mask
chi2_masked[mask == 0] = np.nan
prob = prob * mask
prob[np.isnan(prob)] = 0
print(prob.mean(), prob.max(), prob.min())

# 1D likelihood of sqrt(n)/T
num_bins = 20
counts_noweight, bins = np.histogram(grid, bins=num_bins, range=(-2,2), weights=None, density=True)
counts_weighted, bins = np.histogram(grid, bins=num_bins, range=(-2,2), weights=prob)
counts_norm = np.nan_to_num(counts_weighted / counts_noweight)
counts = np.array((counts_noweight, counts_weighted, counts_norm))
titles = np.array(('uniformly-weighted','probability-weighted','normalized'))

plt.figure(figsize=(8,3))
plt.suptitle(source+' (x,y) = ('+str(idx_x)+','+str(idx_y)+')')
for i in range(3):
	plt.subplot(1,3,i+1)
	plt.title(titles[i])
	plt.tick_params(axis="y", labelleft=False)
	plt.hist(bins[:-1], bins, weights=counts[i], log=False, histtype='step', color='k')
	plt.xlabel(r'$\log(\sqrt{n}/T_k)$')
plt.subplots_adjust(wspace=0.3)
plt.savefig(sou_model+'prob1d_sqrt_n_over_T_'+source+'_'+str(idx_x)+'_'+str(idx_y)+'.pdf', bbox_inches='tight', pad_inches=0.1)
plt.show()

