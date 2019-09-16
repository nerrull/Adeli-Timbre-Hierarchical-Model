from matplotlib import pyplot as plt
import numpy as np
from timeit import default_timer as timer

from utils.file_utils import LibrosaSoundFileLoader
from filterbank.nsgt import NSGT_ERB, upsample_channels
from filterbank.filterbank_operations import extract_envelope

filename = '../data/bass_electronic_018-023-075.wav'
np.random.seed(666)

# Load the file
num_seconds = 16
segment_offset_seconds = 0
sf = LibrosaSoundFileLoader(filename)
sf.get_segment(num_seconds, 0)
sr= sf.get_sample_rate()
signal = sf.get_segment(num_seconds, segment_offset_seconds)

sample_rate = 2048
signal = np.random.rand(sample_rate*num_seconds) -0.5
filterbank = NSGT_ERB(sample_rate, sample_rate*num_seconds,1,plot=False)

start = timer()
c = filterbank.forward(signal)
env = extract_envelope(c)
upsampled_env = np.array(upsample_channels(env))
upsampled_env = upsampled_env/np.max(upsampled_env, axis=1).reshape((-1,1))
print ("Reduced scale forward pass took {} seconds".format( timer()-start))

start = timer()
c2 = filterbank.forward_full_temp(signal)
full_env = extract_envelope(c2)
full_env= full_env/np.max(full_env, axis=1).reshape((-1,1))
print ("Full scale forward pass took {} seconds".format( timer()-start))

for index in range(0, c2.shape[0]):
    plt.figure()
    plt.title("Index {}".format(index ))
    plt.subplot(3,1,1)
    plt.plot(upsampled_env[index], label = "upsampled")
    plt.plot(full_env[index], label = "full")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(env[index])
    plt.subplot(3,1,3)
    plt.plot(full_env[index])

    plt.show()