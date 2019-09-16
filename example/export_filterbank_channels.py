import numpy as np
import struct
import matplotlib.pyplot as plt

from filterbank.nsgt import NSGT_ERB
from filterbank.filterbank_operations import get_lateral_inhibition_coefficients, apply_inhibition, half_wave_rectify, remove_DC_channels, extract_envelope, downsample, compress_coefficients
from utils.file_utils import export_multichannel, loadFile

#Processes an audio file with the filterbank and outputs a multichannel audio
# file where each channel contains the sound information from a singl filter
# band in the filterbank

#In/out settings
filename = '../data/bass_electronic_018-023-075.wav'
out_filename = "out.wav"

# Number of filters per erb band
v =1

#Number of seconds of the sample to process
num_seconds = 4
#Offset from the start of the sample
segment_offset_seconds = 0

#Plot envelopes
plotInhibited = False


#Load and read the file
wr = loadFile(filename)
fs = wr.getframerate()
sampleWidth = wr.getsampwidth()
nframes = wr.getnframes()
nchannels = wr.getnchannels()

segment_offset = int(segment_offset_seconds*fs)
segment_length = int(num_seconds*fs)
discard = wr.readframes(segment_offset)
seg = wr.readframes(segment_length)
num_h =  int(sampleWidth*segment_length/2)
segment = struct.unpack('h'*num_h, seg)

#get first channel
segment =segment[0::nchannels]
signal = segment
Ls = segment_length



nsgt = NSGT_ERB( fs, Ls, v, plot=False)
c = nsgt.forward_full_temp(signal)
c = np.array(c)

export_multichannel(c, wr.getframerate(), './out/'+out_filename)



if plotInhibited:
    envelopes = extract_envelope(c)
    envelopes = downsample(envelopes, 10)
    envelopes = compress_coefficients(envelopes)

    wins = nsgt.get_frequency_windows(plot=False)
    inhibitionCoefs = get_lateral_inhibition_coefficients(wins, plot=False)
    inhibitedEnvelopes = apply_inhibition(envelopes, inhibitionCoefs)
    rectified = half_wave_rectify(inhibitedEnvelopes)
    noDCChannels = remove_DC_channels(rectified)


    plt.figure()
    plt.subplot(221)
    plt.imshow(np.real(c), aspect=float(c.shape[1]) / c.shape[0] * 0.5,
               interpolation='nearest')
    plt.colorbar()
    plt.title("Real part of inverse transform")

    plt.subplot(222)
    plt.imshow(inhibitedEnvelopes, aspect=float(inhibitedEnvelopes.shape[1]) / inhibitedEnvelopes.shape[0] * 0.5,
               interpolation='nearest')
    plt.colorbar()

    plt.title("Hilbert transformed envelopes")

    plt.subplot(223)
    plt.imshow(rectified, aspect=float(rectified.shape[1]) / rectified.shape[0] * 0.5,
               interpolation='nearest')
    plt.colorbar()

    plt.title("Inhibited + rectified envelopes")

    plt.subplot(224)
    plt.imshow(noDCChannels, aspect=float(noDCChannels.shape[1]) / noDCChannels.shape[0] * 0.5,
               interpolation='nearest')
    plt.title("Envelopes with DC channels removed")
    plt.colorbar()
    plt.show()