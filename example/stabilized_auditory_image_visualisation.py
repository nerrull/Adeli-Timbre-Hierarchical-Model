from utils.windows import Windower
from filterbank.two_stage_filterbank import TwoStageFilterBank
from utils.file_utils import SoundFileLoader, LibrosaSoundFileLoader
from filterbank.filterbank_statistics import max_trigger_autocorrelation

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from os.path import exists
from os import makedirs

# Generates a stabilized auditory image (SAI) of a sound file

filename = '../data/concat1.wav'
out_dir = "./out/SAI"

#Number of seconds to sample
num_seconds = 12
segment_offset_seconds = 0

#Parameters for tukey window
slice_length = 2048
window_transition_area = slice_length//4
assert (window_transition_area <slice_length)

#SAI parameters
sample_width = 512
assert (sample_width <slice_length)

#Filters per erb band
erb_filters = 2

save_images = True
if save_images:
    if not exists(out_dir):
        makedirs(out_dir)

# Load the file
sf = LibrosaSoundFileLoader(filename)
signal = np.array(sf.get_segment(num_seconds, segment_offset_seconds))
frame_duration = slice_length/sf.get_sample_rate()
frame_rate  =1/frame_duration
frame_duration_ms = frame_duration*1000

print ("framerate is: {}".format(frame_rate))
filterbank_sliced = TwoStageFilterBank(
    first_stage_downsample=1, v_first_stage=erb_filters)
filterbank_sliced.generateFilterbanks(slice_length, sample_rate=sf.get_sample_rate())
windower_sliced = Windower(slice_length, transition_area=window_transition_area, stride=slice_length)

ims = []
for i, slice in enumerate(windower_sliced.forward(signal)):
    print("Processing slice {}/{}".format(i, len(signal)//slice_length))
    filterbank_sliced.forward_first_stage(slice)
    sai =np.array([max_trigger_autocorrelation(envelope, sample_width) for
                         envelope in filterbank_sliced.raw_envelopes])
    im = plt.imshow(sai, aspect=float(sai.shape[1]) / sai.shape[0] * 0.5,
               interpolation='nearest')
    #Todo : make x label mean something
    ims.append([im])
    if (save_images):
        plt.imsave("./out/SAI/{}.png".format(i), sai)



fig = plt.figure()
ani = anim.ArtistAnimation(fig, ims, interval=frame_duration_ms, blit=True,
                                repeat_delay=1000)
plt.show()


