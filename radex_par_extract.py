import numpy as np
from joblib import Parallel, delayed
import time

start_time = time.time()

# Parameter Settings
par = 'Tex'
molecule_0 = 'co'
molecule_1 = '13co'
molecule_2 = 'c18o'
model = '5d_coarse2'

sou_model = 'radex_model/'
num_cores = 20

linewidth = 15
Nco = np.arange(15.,20.1,0.2)
Tkin = np.arange(1.,2.8,0.1)
nH2 = np.arange(2.,5.1,0.2)
X_13co = np.arange(10,205,10)
X_c18o = np.arange(2,21,1)
round_dens = 1
round_temp = 1

# Pre-processing
incr_dens = round(Nco[1] - Nco[0],1)
incr_temp = round(Tkin[1] - Tkin[0],1)
co_dex = np.round(10**np.arange(0.,1.,incr_dens), 4)
Tk_dex = np.round(10**np.arange(0.,1.,incr_temp), 4)
num_Nco = Nco.shape[0]
num_Tk = Tkin.shape[0]
num_nH2 = nH2.shape[0]
factors_13co = 1./X_13co  
factors_c18o = 1./X_c18o
cycle_dens = co_dex.shape[0]
cycle_temp = Tk_dex.shape[0]
num_X12to13 = X_13co.shape[0]
num_X13to18 = X_c18o.shape[0]
diff_Tk = Tkin[1] - Tkin[0]

def radex_par(i,j,k,m,n):
    powj = j//cycle_dens + int(nH2[0])
    Tk = str(round(diff_Tk*i + int(Tkin[0]), round_temp))
    n_h2 = str(powj)
    N_co = str(k//cycle_dens + int(Nco[0]))
    
    prej_r = str(round(co_dex[j%cycle_dens], round_dens))   
    x13co = str(X_13co[m])
    xc18o = str(X_c18o[n])
    codex = str(round(co_dex[k%cycle_dens], round_dens))
    
    # Change the index here to ensure extracting the desired parameter (e.g. Tex=6, tau=7)
    par_0 = np.genfromtxt('output_'+model+'_'+molecule_0+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'.out', skip_header=13)[:,6]
    par_1 = np.genfromtxt('output_'+model+'_'+molecule_1+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'.out', skip_header=13)[:,6]
    par_2 = np.genfromtxt('output_'+model+'_'+molecule_2+'/'+Tk+'_'+prej_r+'e'+n_h2+'_'+codex+'e'+N_co+'_'+x13co+'_'+xc18o+'.out', skip_header=13)[:,6]
    
    return k,i,j,m,n,par_0,par_1,par_2

# Construct 3D - 5D parameter grids
results = Parallel(n_jobs=num_cores, verbose=5)(delayed(radex_par)(i,j,k,m,n) for n in range(0,num_X13to18) for m in range(0,num_X12to13) for k in range(num_Nco) for j in range(num_nH2) for i in range(num_Tk))

par_co10 =  np.full((num_Nco,num_Tk,num_nH2),np.nan)
par_co21 = np.full((num_Nco,num_Tk,num_nH2),np.nan)
par_13co21 = np.full((num_Nco,num_Tk,num_nH2,num_X12to13),np.nan)
par_13co32 = np.full((num_Nco,num_Tk,num_nH2,num_X12to13),np.nan)
par_c18o21 = np.full((num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18),np.nan)
par_c18o32 = np.full((num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18),np.nan)

for result in results:
    k, i, j, m, n, par_0, par_1, par_2 = result
    par_co10[k,i,j] = par_0[0]
    par_co21[k,i,j] = par_0[1]
    par_13co21[k,i,j,m] = par_1[1]
    par_13co32[k,i,j,m] = par_1[2]
    par_c18o21[k,i,j,m,n] = par_2[1]
    par_c18o32[k,i,j,m,n] = par_2[2]

np.save(sou_model+par+'_'+model+'_co10.npy', par_co10)
np.save(sou_model+par+'_'+model+'_co21.npy', par_co21)
np.save(sou_model+par+'_'+model+'_13co21.npy', par_13co21)
np.save(sou_model+par+'_'+model+'_13co32.npy', par_13co32)
np.save(sou_model+par+'_'+model+'_c18o21.npy', par_c18o21)
np.save(sou_model+par+'_'+model+'_c18o32.npy', par_c18o32) 
print(par + ' models saved; elapsed time for construction: %s sec' % ((time.time() - start_time)))

# Construct 5D parameter grids for 12CO and 13CO
new_model = '6d_coarse2'
num_Nco = 26
num_Tk = 18
num_nH2 = 16
num_X12to13 = 20
num_X13to18 = 19
num_beamfill = 14  #np.arange(-1.3, 0.1, 0.1)

temp = np.repeat(par_co21[:, :, :, np.newaxis], num_X12to13, axis=3)
par_co21_5d = np.repeat(temp[:, :, :, :, np.newaxis], num_X13to18, axis=4)
temp2 = np.repeat(par_co10[:, :, :, np.newaxis], num_X12to13, axis=3)
par_co10_5d = np.repeat(temp2[:, :, :, :, np.newaxis], num_X13to18, axis=4)
par_13co21_5d = np.repeat(par_13co21[:, :, :, :, np.newaxis], num_X13to18, axis=4)
par_13co32_5d = np.repeat(par_13co32[:, :, :, :, np.newaxis], num_X13to18, axis=4) 

# Construct 6d parameter grids from 5d by adding the beam filling factor dimension
par_co10_6d = np.repeat(par_co10_5d[:, :, :, :, :, np.newaxis], num_beamfill, axis=5)
par_co21_6d = np.repeat(par_co21_5d[:, :, :, :, :, np.newaxis], num_beamfill, axis=5)
par_13co21_6d = np.repeat(par_13co21_5d[:, :, :, :, :, np.newaxis], num_beamfill, axis=5)
par_13co32_6d = np.repeat(par_13co32_5d[:, :, :, :, :, np.newaxis], num_beamfill, axis=5)
par_c18o21_6d = np.repeat(par_c18o21[:, :, :, :, :, np.newaxis], num_beamfill, axis=5)
par_c18o32_6d = np.repeat(par_c18o32[:, :, :, :, :, np.newaxis], num_beamfill, axis=5)

np.save(sou_model+par+'_'+new_model+'_co10.npy', par_co10_6d)
np.save(sou_model+par+'_'+new_model+'_co21.npy', par_co21_6d)
np.save(sou_model+par+'_'+new_model+'_13co21.npy', par_13co21_6d)
np.save(sou_model+par+'_'+new_model+'_13co32.npy', par_13co32_6d)
np.save(sou_model+par+'_'+new_model+'_c18o21.npy', par_c18o21_6d)
np.save(sou_model+par+'_'+new_model+'_c18o32.npy', par_c18o32_6d)

print('6D '+par+' grids saved.')

end_time = time.time()
print('Total lapsed time: %s sec' % ((end_time - start_time)))
