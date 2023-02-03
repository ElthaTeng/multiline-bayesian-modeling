import numpy as np

# Parameter Settings
sou_model = 'radex_model/'
base_model = '5d_coarse2'
model = '6d_coarse2'
num_Nco = 26
num_Tk = 18
num_nH2 = 16
num_X12to13 = 20
num_X13to18 = 19
beam_fill = 10**np.arange(-1.3, 0.1, 0.1)

# import flux data
flux_co10 = np.load(sou_model+'flux_'+base_model+'_co10.npy')
flux_co21 = np.load(sou_model+'flux_'+base_model+'_co21.npy')
flux_co32 = np.load(sou_model+'flux_'+base_model+'_co32.npy') 
flux_13co21 = np.load(sou_model+'flux_'+base_model+'_13co21.npy')
flux_13co32 = np.load(sou_model+'flux_'+base_model+'_13co32.npy')
flux_c18o21_5d = np.load(sou_model+'flux_'+base_model+'_c18o21.npy')
flux_c18o32_5d = np.load(sou_model+'flux_'+base_model+'_c18o32.npy')
flux_13co10 = np.load(sou_model+'flux_'+base_model+'_13co10.npy')
flux_c18o10_5d = np.load(sou_model+'flux_'+base_model+'_c18o10.npy')

temp = np.repeat(flux_co21[:, :, :, np.newaxis], num_X12to13, axis=3)
flux_co21_5d = np.repeat(temp[:, :, :, :, np.newaxis], num_X13to18, axis=4)
temp2 = np.repeat(flux_co10[:, :, :, np.newaxis], num_X12to13, axis=3)
flux_co10_5d = np.repeat(temp2[:, :, :, :, np.newaxis], num_X13to18, axis=4)
temp3 = np.repeat(flux_co32[:, :, :, np.newaxis], num_X12to13, axis=3)                                                      
flux_co32_5d = np.repeat(temp3[:, :, :, :, np.newaxis], num_X13to18, axis=4) 
flux_13co21_5d = np.repeat(flux_13co21[:, :, :, :, np.newaxis], num_X13to18, axis=4)
flux_13co32_5d = np.repeat(flux_13co32[:, :, :, :, np.newaxis], num_X13to18, axis=4) 

flux_13co10_5d = np.repeat(flux_13co10[:, :, :, :, np.newaxis], num_X13to18, axis=4) 

# Construct 6d flux models from 5d by adding the beam filling factor dimension
flux_co10_6d = flux_co10_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_co21_6d = flux_co21_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_co32_6d = flux_co32_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0]) 
flux_13co21_6d = flux_13co21_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_13co32_6d = flux_13co32_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_c18o21_6d = flux_c18o21_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_c18o32_6d = flux_c18o32_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])

flux_13co10_6d = flux_13co10_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])
flux_c18o10_6d = flux_c18o10_5d.reshape(num_Nco,num_Tk,num_nH2,num_X12to13,num_X13to18,1) * beam_fill.reshape(1,1,1,1,1,beam_fill.shape[0])

np.save(sou_model+'flux_'+model+'_co10.npy', flux_co10_6d)
np.save(sou_model+'flux_'+model+'_co21.npy', flux_co21_6d)
np.save(sou_model+'flux_'+model+'_13co21.npy', flux_13co21_6d)
np.save(sou_model+'flux_'+model+'_13co32.npy', flux_13co32_6d)
np.save(sou_model+'flux_'+model+'_c18o21.npy', flux_c18o21_6d)
np.save(sou_model+'flux_'+model+'_c18o32.npy', flux_c18o32_6d)

np.save(sou_model+'flux_'+model+'_13co10.npy', flux_13co10_6d)
np.save(sou_model+'flux_'+model+'_c18o10.npy', flux_c18o10_6d)
np.save(sou_model+'flux_'+model+'_co32.npy', flux_co32_6d)

