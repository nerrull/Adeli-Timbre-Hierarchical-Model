from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as MP
from scipy.interpolate import interp1d

from .erbletwin import  adeliErbletWin, calcErbWinRange
from utils.fft import fftp, ifftp
from utils.util import chkM, arrange,chnmap
# from filterbank.nsgt_opt import nsgtf_loop
class NSGT_ERB:
    def __init__(self, sample_rate, frame_length, filters_per_band, cutoff_frequency = None, plot = False, dtype = np.float64 ):
        assert sample_rate>0
        assert frame_length>0

        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.dtype = dtype

        #Create filterbanks in frequency domain
        self.g, rfbas, self.M, self.center_frequencies = adeliErbletWin(filters_per_band, sample_rate, frame_length, plot=False, max_freq=cutoff_frequency, dtype= dtype)
        self.windows, self.nn = calcErbWinRange(self.g, rfbas, frame_length)
        self.loopparams = self.generate_loop_params()

    def get_windows(self):
        return self.g, self.windows, self.frame_length, self.sample_rate

    def generate_loop_params(self):
        loopparams = []
        for mii, gii, win_range in zip(self.M, self.g, self.windows):
            Lg = len(gii)
            col = int(ceil(float(Lg) / mii))
            assert col * mii >= Lg
            gi1 = gii[:(Lg + 1) // 2]
            gi2 = gii[-(Lg // 2):]
            p = (mii, gii, gi1, gi2, win_range, Lg, col)
            loopparams.append(p)
        return loopparams

    def forward(self, s):
        fft = fftp(measure=False, dtype=self.dtype)
        ifft = ifftp(measure=False, dtype=self.dtype)

        Ls = len(s)
        ft = fft(s)
        temp0 = np.zeros(Ls, dtype=ft.dtype)
        if self.nn > Ls:
            ft = np.concatenate((ft, np.zeros(self.nn - Ls, dtype=ft.dtype)))

        c = nsgtf_loop(self.loopparams, ft, temp0, False)
        # Single threaded map(ifft, c)
        # multiprocessing (doesnt work) mmap = MP.Pool().map
        outChannel = list(map(ifft, c))
        return outChannel

    def forward_full_temp(self, s):
        fft = fftp(measure=False, dtype=self.dtype)
        ifft = ifftp(measure=False, dtype=self.dtype)

        Ls = len(s)
        ft = fft(s)
        temp0 = np.zeros(Ls, dtype=ft.dtype)
        if self.nn > Ls:
            ft = np.concatenate((ft, np.zeros(self.nn - Ls, dtype=ft.dtype)))

        c = nsgtf_loop(self.loopparams, ft, temp0, True)
        # Single threaded map(ifft, c)
        # multiprocessing (doesnt work) mmap = MP.Pool().map
        # outc =np.array(list(map(fft, c)))
        # out_channel = np.flip(outc ,axis =1)
        out_channel = list(map(ifft, c))
        return np.array(out_channel)

    def get_impulse_response(self):
        signal = np.zeros((self.frame_length), dtype = np.float32)
        signal[self.frame_length//2] =2**16

        response = self.forward(signal)
        clipped_response = []
        for r in response:
            non_zero = np.argwhere(np.real(r).astype(np.int32) != 0)
            clipped_response.append(r[np.min(non_zero) :np.max(non_zero)])

        return clipped_response


    def get_frequency_windows(self, plot=False):
        numWindows = (len(self.windows) - 2) // 2 + 1  # first and last windows are unique. they straddle the limits
        firstWindow = self.g[0]
        lastWindow = self.g[numWindows]
        wins = []

        temp = np.zeros((self.frame_length,))
        temp[self.windows[0]] = np.fft.fftshift(firstWindow)
        wins.append(temp)

        for i in range(1, numWindows):
            winRange1 = self.windows[i]
            winRange2 = self.windows[-i]
            temp = np.zeros((self.frame_length,))
            temp[winRange1] = np.fft.fftshift(self.g[i])
            temp[winRange2] = np.fft.fftshift(self.g[-i])
            diff = len(winRange1) - len(winRange2)
            wins.append(temp)

        temp = np.zeros((self.frame_length,))
        temp[self.windows[numWindows]] = np.fft.fftshift(lastWindow)
        wins.append(temp)

        if plot:
            f = plt.figure()
            for w in wins[:]:
                plt.plot(w)
            plt.show()
        return wins


#Could use something like this to upsample (much more efficient) :
# https://www.dsprelated.com/showarticle/1123.php
# Needs to have a discrete upsampling factor though
def upsample_channels(channels, max_length =None):
    if max_length ==None:
        max_length = channels[-1].shape[0]

    up_channels = []
    for c in channels:
        channel_length = len(c)
        lerp =interp1d(np.arange(0, channel_length), c)
        steps = np.arange(0, max_length)*float(channel_length-1)/float(
            max_length)
        interp = lerp(steps)
        up_channels.append(interp)
    return np.array(up_channels)

def nsgtf_loop(loopparams, ft, temp0, k_t_r):
    c =[]
    N = len(loopparams)
    Pos_Chans_No = (N ) // 2

    #First and last channels don't have mirror filters
    c.append(nsgtf(ft,loopparams[0], temp0, k_t_r))

    for i in range(1, Pos_Chans_No ):
        positiveParams = loopparams[i]
        negativeParams = loopparams[N-i]
        if k_t_r:
            c.append( nsgtf(ft, positiveParams, temp0, k_t_r)+  nsgtf(ft,negativeParams,temp0, k_t_r))
        else:
            c.append( np.append(nsgtf(ft, positiveParams, temp0, k_t_r),
                                nsgtf(ft,negativeParams,temp0, k_t_r)))
    #First and last channels don't have mirrored filters
    c.append(nsgtf(ft,loopparams[Pos_Chans_No], temp0, k_t_r))
    return c

def nsgtf(ft, params, temp0, keep_temporal_resolution =True):
    # type: (object, object, object) -> object
    temp = temp0.copy()
    (mii,_, gi1, gi2, win_range, Lg, col) = params

    # modified version to avoid superfluous memory allocation
    t = temp[win_range]
    t1 = t[:(Lg + 1) // 2]
    t1[:] = gi2  # if mii is odd, this is of length mii-mii//2
    t2 = t[-(Lg // 2):]
    t2[:] = gi1  # if mii is odd, this is of length mii//2

    ftw = ft[win_range]
    t1 *= ftw[:Lg // 2]
    t2 *= ftw[Lg // 2:]

    if not keep_temporal_resolution:
        return t

    temp[win_range] = t

    if col >1:
        temp = np.sum(temp.reshape((mii, -1)), axis=1)
    else:
        temp = temp.copy()
    return temp
