
import numpy as np
from math import exp, floor, ceil, pi
from librosa.filters import get_window

def hannwin(l):
    r = np.arange(l, dtype=float)
    r *= np.pi * 2. / l
    r = np.cos(r)
    r += 1.
    r *= 0.5
    return r



def blackharr(n, l=None, mod=True):
    if l is None:
        l = n
    nn = (n // 2) * 2
    k = np.arange(n)
    if not mod:
        bh = 0.35875 - 0.48829 * np.cos(k * (2 * pi / nn)) + 0.14128 * np.cos(k * (4 * pi / nn)) - 0.01168 * np.cos(
            k * (6 * pi / nn))
    else:
        bh = 0.35872 - 0.48832 * np.cos(k * (2 * pi / nn)) + 0.14128 * np.cos(k * (4 * pi / nn)) - 0.01168 * np.cos(
            k * (6 * pi / nn))
    bh = np.hstack((bh, np.zeros(l - n, dtype=bh.dtype)))
    bh = np.hstack((bh[-n // 2:], bh[:-n // 2]))
    return bh


def blackharrcw(bandwidth, corr_shift):
    flip = -1 if corr_shift < 0 else 1
    corr_shift *= flip

    M = np.ceil(bandwidth / 2 + corr_shift - 1) * 2
    win = np.concatenate((np.arange(M // 2, M), np.arange(0, M // 2))) - corr_shift
    win = (0.35872 - 0.48832 * np.cos(win * (2 * np.pi / bandwidth)) + 0.14128 * np.cos(
        win * (4 * np.pi / bandwidth)) - 0.01168 * np.cos(win * (6 * np.pi / bandwidth))) * (win <= bandwidth) * (
          win >= 0)

    return win[::flip], M


def cont_tukey_win(n, sl_len, tr_area):
    g = np.arange(n) * (sl_len / float(n))
    g[np.logical_or(g < sl_len / 4. - tr_area / 2., g > 3 * sl_len / 4. + tr_area / 2.)] = 0.
    g[np.logical_and(g > sl_len / 4. + tr_area / 2., g < 3 * sl_len / 4. - tr_area / 2.)] = 1.
    #
    idxs = np.logical_and(g >= sl_len / 4. - tr_area / 2., g <= sl_len / 4. + tr_area / 2.)
    temp = g[idxs]
    temp -= sl_len / 4. + tr_area / 2.
    temp *= pi / tr_area
    g[idxs] = np.cos(temp) * 0.5 + 0.5
    #
    idxs = np.logical_and(g >= 3 * sl_len / 4. - tr_area / 2., g <= 3 * sl_len / 4. + tr_area / 2.)
    temp = g[idxs]
    temp += -3 * sl_len / 4. + tr_area / 2.
    temp *= pi / tr_area
    g[idxs] = np.cos(temp) * 0.5 + 0.5
    #
    return g


def tukey_win(n, tr_area):
    g = np.arange(n, dtype=np.float32)
    out =  np.zeros(n, dtype = np.float32)
    out[np.logical_or(g < tr_area , g >n - tr_area )] = 0.
    out[np.logical_and(g >= tr_area , g <= n - tr_area )] = 1.
    #
    idxs = g <= tr_area
    temp = g[idxs] - tr_area
    #temp -= sl_len / 4. + tr_area / 2.
    temp *= pi / tr_area
    out[idxs] = np.cos(temp) * 0.5 + 0.5
    # #
    idxs =g >= (n -tr_area)
    temp = g[idxs] -tr_area
     #temp += -3 * sl_len / 4. + tr_area / 2.
    temp *= pi / tr_area
    out[idxs] = np.cos(temp) * 0.5 + 0.5
    #
    return out

def tgauss(ess_ln, ln=0):
    if ln < ess_ln:
        ln = ess_ln
    #
    g = np.zeros(ln, dtype=float)
    sl1 = int(floor(ess_ln / 2))
    sl2 = int(ceil(ess_ln / 2)) + 1
    r = np.arange(-sl1, sl2)  # (-floor(ess_len/2):ceil(ess_len/2)-1)
    r = np.exp((r * (3.8 / ess_ln)) ** 2 * -pi)
    r -= exp(-pi * 1.9 ** 2)
    #
    g[-sl1:] = r[:sl1]
    g[:sl2] = r[-sl2:]
    return g

class Windower():
    def __init__(self, length,  transition_area=None, stride= None, amplitude=1.):
        self.l = length
        self.stride = int(stride)
        self.transition_area = transition_area
        if stride ==None:
            self.stride = length//2
        if transition_area == None:
            self.transition_area = length // 2
        self.amplitude = amplitude
        self.last_frame = None

        self.w = tukey_win(self.l, self.transition_area)

    def forward(self, signal):
        signal_len = len(signal)
        num_windows = int(floor(signal_len/self.stride))
        for i in range(num_windows):
            step = i*self.stride
            if (step +self.l) >signal_len:
                pt_1 = signal[step:]
                ret = np.concatenate((pt_1, np.zeros((self.l - len(pt_1)))))*self.w
            else :
                ret = signal[step: step + self.l] * self.w

            yield ret

        if signal_len % self.stride !=0 :
            print("IS THIS EVER CALLED?")
            pt_1 = signal[self.stride*num_windows :]
            yield np.concatenate((pt_1,  np.zeros((self.l -len(pt_1))) ))

    def recombine(self, frame):
        if self.last_frame is None :
            retval = frame[:,0:self.stride]
            frame = frame*self.w
        else:
            frame = frame*self.w
            retval = self.last_frame[:,self.stride:]+ frame [:,0:self.stride]

        self.last_frame =frame

        return retval


class HannWindower():
    def __init__(self, length, stride= None, amplitude=1.):
        self.l = length
        self.stride = int(stride)
        if stride ==None:
            self.stride = length//2

        self.amplitude = amplitude
        self.last_frame = None

        self.w = get_window('hann', self.l, True)

    def forward(self, signal):
        signal_len = len(signal)
        num_windows = int(floor(signal_len/self.stride))
        for i in range(num_windows):
            step = i*self.stride
            if (step +self.l) >signal_len:
                pt_1 = signal[step:]
                ret = np.concatenate((pt_1, np.zeros((self.l - len(pt_1)))))*self.w
            else :
                ret = signal[step: step + self.l] * self.w

            yield ret

        if signal_len % self.stride !=0 :
            print("IS THIS EVER CALLED?")
            pt_1 = signal[self.stride*num_windows :]
            yield np.concatenate((pt_1,  np.zeros((self.l -len(pt_1))) ))

    def combine(self, frame1,frame2, down_sample_factor):
        l  =len(frame2)
        s = int(self.stride/self.l *l)

        if self.l == self.stride:
            return  np.concatenate((frame1, frame2))

        frame1[-l+s:] += frame2[ 0:(-s)]
        frame1 = np.concatenate((frame1, frame2[-s:]), axis =0)
        return frame1

    def recombine(self, frame):
        if self.last_frame is None :
            retval = frame[:,0:self.stride]
            frame = frame*self.w
        else:
            frame = frame*self.w
            retval = self.last_frame[:,self.stride:]+ frame [:,0:self.stride]

        self.last_frame =frame

        return retval

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    signal_len = 2048
    sl_len = 2048
    tr_area = 512

    conttukey_win = cont_tukey_win(signal_len, sl_len, tr_area)

    tukey_win = tukey_win(signal_len, tr_area)

    spec = np.fft.fft(tukey_win)

    plt.subplot(411)
    plt.plot(tukey_win)
    plt.title("tukey")
    plt.axvline(x=tr_area, color='r')
    plt.axvline(x=signal_len-tr_area,color = 'r' )
    plt.subplot(412)
    plt.plot(np.fft.fftshift(np.abs(spec)**2))
    plt.title("Power spec")
    plt.subplot(413)
    plt.plot(np.angle(spec))
    plt.title("Phase spec")

    plt.show()