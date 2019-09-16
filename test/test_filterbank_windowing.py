from utils.windows import Windower
from filterbank.two_stage_filterbank import TwoStageFilterBank
from utils.file_utils import SoundFileLoader, LibrosaSoundFileLoader
from filterbank.filterbank_statistics import max_trigger_autocorrelation

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np


# Evaluate the effect of windowing the signal

# Compares filterbank output when we process the full signal vs adding and
# concatenating slices

filename = '../data/concat1.wav'
num_seconds = 12
segment_offset_seconds = 0
window_transition_area = 512

# Load the file
sf = LibrosaSoundFileLoader(filename)
signal = np.array(sf.get_segment(num_seconds, segment_offset_seconds))

filterbank_full = TwoStageFilterBank()
filterbank_full.generateFilterbanks(len(signal), sample_rate=sf.get_sample_rate())

windower_full = Windower(len(signal), transition_area=window_transition_area, stride=len(signal))
s = windower_full.forward(signal)

filterbank_full.forward(s.__next__())

slice_length = 2048
filterbank_sliced = TwoStageFilterBank(first_stage_downsample=1)
filterbank_sliced.generateFilterbanks(slice_length, sample_rate=sf.get_sample_rate())
windower_sliced = Windower(slice_length, transition_area=window_transition_area, stride=slice_length)

raws = []
inhs = []
spectral_envs = []
for slice in windower_sliced.forward(signal):
    filterbank_sliced.forward(slice)
    spectral_envs.append(filterbank_sliced.spectral_envelope)
    raws.append(filterbank_sliced.raw_envelopes)
    inhs.append(filterbank_sliced.inhibited_envelopes)

spectral_envs= np.vstack(spectral_envs)

raw_envelopes = filterbank_full.raw_envelopes
inhibited_envelopes = filterbank_full.inhibited_envelopes

f = plt.figure(figsize=(20, 10))
ax = f.add_subplot(221)
im = ax.imshow(raw_envelopes, aspect=float(raw_envelopes.shape[1]) / raw_envelopes.shape[0] * 0.5,
               interpolation='nearest')
ax.set_title("Full signal Compressed Envelopes")

ax = f.add_subplot(222)
im = ax.imshow(inhibited_envelopes, aspect=float(inhibited_envelopes.shape[1]) / inhibited_envelopes.shape[0] * 0.5,
               interpolation='nearest')
ax.set_title("Full signal Inhibited and rectified")


raw_envelopes = np.hstack(raws)
inhibited_envelopes = np.hstack(inhs)

ax = f.add_subplot(223)
im = ax.imshow(raw_envelopes, aspect=float(raw_envelopes.shape[1]) / raw_envelopes.shape[0] * 0.5,
               interpolation='nearest')
ax.set_title("Full signal Compressed Envelopes")


ax.set_title("Full signal Inhibited and rectified")



#Get average channel energy (spectral envelope
spectral_env_full = filterbank_full.spectral_envelope
spectral_env_split = np.sum(spectral_envs, axis=0)

#Normalize
spectral_env_full = spectral_env_full/np.max(spectral_env_full)
spectral_env_sliced = spectral_env_split/np.max(spectral_env_split)

# Compare
fig_rows= 4
fig_cols= 3

f = plt.figure(figsize=(20, 10))
ax = f.add_subplot(fig_rows,fig_cols,1)
ax.plot(spectral_env_full)
ax.set_title("Full signal normalized spectral envelope Envelopes")

ax.plot(spectral_env_sliced)
ax.set_title("Sliced signal normalized spectral envelope Envelopes")

ax = f.add_subplot(fig_rows,fig_cols,fig_cols*1 +1)
im = ax.imshow(spectral_envs.T,interpolation='nearest')
ax.set_title("Sliced Spectral envelope over time")
plt.colorbar(im,ax=ax)

cum = np.cumsum(spectral_envs.T, axis = 1)
cum_norm = cum/np.max(cum, axis=0)
ax = f.add_subplot(fig_rows,fig_cols,fig_cols*1 +2)
im = ax.imshow(cum_norm,interpolation='nearest')
ax.set_title("Sliced Spectral cum_sum over time")
plt.colorbar(im,ax=ax)


diff = np.diff(cum_norm, axis=1)
ax = f.add_subplot(fig_rows,fig_cols,fig_cols*1 +3)

im = ax.imshow(diff,interpolation='nearest')
ax.set_title("Sliced Spectral  diff over time")
plt.colorbar(im,ax=ax)


diff_full = spectral_envs.T/np.max(spectral_envs) - \
            np.repeat(spectral_env_full.reshape(spectral_env_full.shape[0], 1), spectral_envs.shape[0], axis =1)
ax = f.add_subplot(fig_rows,fig_cols,fig_cols*2 +1)
im = ax.imshow(diff_full,interpolation='nearest')
ax.set_title("Difference from average specral envelope over time")
plt.colorbar(im,ax=ax)

ax = f.add_subplot(fig_rows,fig_cols,fig_cols*3 +1)
ax.plot(np.sum(diff_full, axis =1))
ax.set_title("Total difference from average spectral envelope over time")
plt.colorbar(im,ax=ax)

diff_full = cum_norm - \
            np.repeat(spectral_env_full.reshape(spectral_env_full.shape[0], 1), spectral_envs.shape[0], axis =1)
ax = f.add_subplot(fig_rows,fig_cols,fig_cols*2 +2)
im = ax.imshow(diff_full,interpolation='nearest')
ax.set_title("Difference from full specral envelope for time-averaged sliced spectral envelope")
plt.colorbar(im,ax=ax)

ax = f.add_subplot(fig_rows,fig_cols,fig_cols*3 +2)
ax.plot(np.sum(diff_full, axis =1))
ax.set_title("Difference from average specral envelope for time-averaged sliced spectral envelope")
plt.colorbar(im,ax=ax)
# ax = f.add_subplot(223)
# im = ax.imshow(raw_envelopes, aspect=float(raw_envelopes.shape[1]) / raw_envelopes.shape[0] * 0.5,
#                interpolation='nearest')
# ax.set_title("Full signal Compressed Envelopes")

# ax.plot(spectral_env_full- spectral_env_sliced)
# ax.set_title("Difference normalized spectral envelope Envelopes")


plt.show()
