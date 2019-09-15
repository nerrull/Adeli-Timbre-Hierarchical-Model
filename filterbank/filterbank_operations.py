import numpy as np
import scipy.signal as scignal
import matplotlib.pyplot as plt

def compress_coefficients(c):
    for i, row in enumerate(c):
        c[i] = np.sqrt(row+1)-1
    return c

'''
 find filter correlations in freq domain
        WINCor=zeros(Pos_Chans_No,Pos_Chans_No);
        for i=1:Pos_Chans_No
            if i<=Pos_Chans_No-1
                WINCor(i,i+1)=FILTS(:,i)'*FILTS(:,i+1);
                WINCor(i+1,i)=WINCor(i,i+1);
            end
        end
        WSRvec=sum(WINCor,2);
        WINCor=WINCor./repmat(WSRvec,1,Pos_Chans_No);
'''
def get_lateral_inhibition_coefficients(filterEnvelopes, plot=False):
    numFilters = len(filterEnvelopes)
    filterCorrelations = np.zeros((numFilters,numFilters))
    for index in range(0, len(filterEnvelopes)-1):
        filterCorrelations[index, index+1]= np.dot(filterEnvelopes[index],filterEnvelopes[index+1])
        filterCorrelations[index+1, index] = filterCorrelations[index, index+1]

    row_sums = np.sum(filterCorrelations, axis=1)
    filterCorrelations = filterCorrelations/row_sums[:, np.newaxis]

    if plot:
        plt.figure()
        plt.imshow(filterCorrelations)
        plt.show()
    return filterCorrelations


#% Apply lateral inhibition
def apply_inhibition(envelopes, coefficients):
    inhibition = np.matmul(coefficients, envelopes )
    return np.subtract(envelopes, inhibition)

def half_wave_rectify (envelopes):
    return (envelopes + np.abs(envelopes))/2

#  ENVS_LI=env(No_DC_chs+1:end,:); % 2 channels removed
def remove_DC_channels(envelopes):
    return envelopes[2:-1]

#Take the absolute value of the hilbert transform of the real part of the
# signal to extract the envelope
def extract_envelope(out_channel, multithreading = False):
    hilbert_channels = []
    for channel in out_channel:
        hilbert_channels.append(np.abs(scignal.hilbert(np.real(channel))))
    return np.array(hilbert_channels)

#Take the absolute value of the hilbert transform of the real part of the
# signal to extract the envelope
def extract_envelope_single_channel(channel, multithreading = False):
    return np.abs(scignal.hilbert(np.real(channel)))

def downsample(array, factor):
    return array[:,::factor]

def rms(values):
    return np.sqrt(1 /values.shape[-1] * np.sum(np.square(values), axis=-1))

def get_spectral_envelope(envelopes):
    se = []
    for env in envelopes:
        se.append(rms(env))

    return np.array(se)

def zero_pad(signal, numzeros):
    return np.concatenate([np.zeros(numzeros), signal, np.zeros(numzeros)], axis=0)

def second_stage_amp_mod_correlation(amp_mod):
    num_channels = amp_mod.shape[0]
    correlation = np.zeros(num_channels)
    std = np.std(amp_mod, axis =1)
    ind = np.argwhere(std>0).flatten()
    tmp = np.zeros(num_channels)
    for index in ind:
        if index < num_channels-1 and std[index +1]>0:
            tmp[index] =np.corrcoef(amp_mod[index], amp_mod[index+1])[0,1]
    correlation[0] = tmp[0]
    correlation[-1]=tmp[-1]
    correlation[1:-2] = (tmp[1:-2]+tmp[2:-1])/2
    return correlation.reshape((1,-1))

"""
Roughness depends on the modulation depth of each channel as well as the
correlation between channels
"""
def get_roughness(amp_mod, modulation_depth, sr):
    #ignore first 34 ms of AM signals
    ignore_ms = int(np.ceil(0.034 / (1/sr)))
    correlation = second_stage_amp_mod_correlation(amp_mod)
    corrected_modulation_depth = correlation.dot(modulation_depth)
    gamma = 2 ** .69
    instantaneous_roughness = gamma *np.square(corrected_modulation_depth)[0]

    instantaneous_roughness[0:ignore_ms]=0
    instantaneous_roughness[-ignore_ms:]=0

    effective_roughness= rms(instantaneous_roughness)
    return effective_roughness, instantaneous_roughness
