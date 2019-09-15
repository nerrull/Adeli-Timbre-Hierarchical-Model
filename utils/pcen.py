import scipy
import numpy as np

def pcen_time(E, sr=22050, hop_length=512, alpha=0.98, delta=2, r=0.5, t=0.395, eps=1e-6):
    s = 1 - np.exp(- float(hop_length) / (t * sr))
    M = scipy.signal.lfilter([s], [1, s - 1], E)
    smooth = (eps + M)**(-alpha)
    return (E * smooth + delta)**r - delta**r


def pcen(E, alpha=0.98, delta=2, r=0.5, s=0.025, eps=1e-6):
    M = scipy.signal.lfilter([s], [1, s - 1], E)
    smooth = (eps + M)**(-alpha)
    return (E * smooth + delta)**r - delta**r