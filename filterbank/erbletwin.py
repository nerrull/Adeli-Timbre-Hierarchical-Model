

import numpy as np
from ltfatpy.fourier.pgauss import pgauss
from utils.plotting import plot_windows, plotCenterFrequencies
from scipy.stats import beta

#TODO : remove dependancy on ltfat pgauss function
# http://ltfat.github.io/doc/fourier/pgauss.html

def hzToErb(hz):
    return 9.2645*np.sign(hz) * np.log(1 +abs(hz)*0.00437)

def erbToHz(erb):
    return (1 / 0.00437) * np.sign(erb) * (np.exp(abs(erb) / 9.2645) - 1)

def getErbBandwidth(frequency):
    return 24.7 + frequency / 9.265

def erbSpace(min_frequency, max_frequency, num_channels):
    erb_numbers = np.linspace(hzToErb(min_frequency), hzToErb(max_frequency/2), num_channels)
    centerFrequencies = erbToHz(erb_numbers)
    return centerFrequencies, erb_numbers

def get_carrier_frequency_gain(V, sr):
    Nf = V * np.ceil(hzToErb(sr / 2) - hzToErb(0)) # Number of freq. channels
    fc, erb_numbers = erbSpace(0, sr/2 , Nf)
    beta.pdf(erb_numbers/50, 1.75, 2.65)
    g = beta.pdf(erb_numbers / 50, 1.75, 2.65)
    return g / np.max(g)

#based on erbletwin.m from erblet transform code in LTFAT
def erbletwin(V, sr, Ls, dtype=np.float64):

    freqRes = sr/Ls #frequency resolution
    print("Frequency resolution {}".format(freqRes))
    fmin = 0
    fmax= sr/2
    erbMax= hzToErb(fmax)
    erbMin = hzToErb(fmin)
    Nf = int(V*np.ceil( erbMax- erbMin) )#number of frequency channels
    erbSpace = np.linspace(erbMin, erbMax, Nf)
    centerFrequencies = erbToHz(erbSpace)
    centerFrequencies = np.append( centerFrequencies, np.flip(centerFrequencies[1:-1],0))
    bandwidths = getErbBandwidth(centerFrequencies)/V

    cfPositions = np.round (centerFrequencies/freqRes)
    cfPositions[Nf:] = Ls - cfPositions[Nf:]+1 # Extension to negative freq. #Todo make sure this step is equivalent to octave code

    shift = np.append(np.mod(-cfPositions[-1], Ls), np.diff(cfPositions)).astype(int)

    Lwin = 4*np.round(bandwidths/freqRes).astype(int)
    #Set all odd Lwin values to even numbers (to avoid indexing errors in windows computation)
    Lwin[np.mod(Lwin,2)!=0] +=1
    M = Lwin

    factor =np.round(1.0/0.79)
    g = []
    for k in range(0, 2*Nf-2):
        gt, tfr = pgauss(Lwin[k].tolist(), width=factor *bandwidths[k]/freqRes)
        gt = gt/np.max(gt);
        g.append(gt)
        #g.append(gt)

    g[1] = 1/np.sqrt(2)*g[1]
    g[-1] = 1/np.sqrt(2)*g[-1]

    return g, shift, M


#Todo : make this support V <1 (multiple erb bandwidths covered by each filter)
#based on erbletwin.m from erblet transform code in LTFAT
def adeliErbletWin( V, sr, Ls,  max_freq = None, plot=False, dtype=np.float64):
    freqRes = sr/float(Ls)#frequency resolution
    fmin = 0
    fmax= sr/2
    if max_freq !=None and max_freq<sr/2:
        fmax = max_freq
    erbMax= hzToErb(fmax)
    erbMin = hzToErb(fmin)
    Nf = int(V*np.ceil( erbMax- erbMin) ) #number of frequency channels
    Nf = 24
    erbSpace = np.linspace(erbMin, erbMax, Nf)
    centerFrequencies = erbToHz(erbSpace)
    cf = centerFrequencies
    centerFrequencies = np.append( centerFrequencies, np.flip(centerFrequencies[1:-1],0))

    bandwidths = getErbBandwidth(centerFrequencies)/V

    cfPositions = np.round (centerFrequencies/freqRes)
    cfPositions[Nf:] = Ls - cfPositions[Nf-2:0:-1] # Extension to negative freq. #Todo make sure this step is equivalent to octave code
    #cfPositions[Nf:] = Ls - cfPositions[Nf:]+1 # Extension to negative freq. #Todo make sure this step is equivalent to octave code
    #cfPositions=np.round(cfPositions)
    shift = np.append(np.mod(-cfPositions[-1], Ls), np.diff(cfPositions)).astype(int)
    Lwin = 4*np.round(bandwidths/freqRes).astype(int)

    #Set all odd Lwin values to even numbers (to avoid indexing errors in windows computation)
    Lwin[np.mod(Lwin,2)!=0] +=1
    M = Lwin

    if plot:
        plotCenterFrequencies(centerFrequencies, cfPositions)


    #gonna comment this out because it tends to work anyway
    for i in range (1, Nf -1):
        if Ls  - cfPositions[i] !=  cfPositions[-i] :
            raise Exception("Filter windows at band {} are not symmetric".format (i))

        if Lwin[i] !=  Lwin[-i] :
            raise Exception("Filter windows at band {} are not symmetric".format (i))

    g = []
    for k in range(0, 2*Nf-2):
        gt, tfr = pgauss(Lwin[k].tolist(), width=np.round(1.0/0.79 *bandwidths[k]/freqRes))
        g.append(gt/np.max(gt))

    return g, shift, M, cf

def calcErbWinRange(g, shift, Ls):
    timepos = np.cumsum(shift)
    nn = timepos[-1]
    timepos -= shift[0]  # Calculate positions from shift vector

    wins = []
    for gii, tpii in zip(g, timepos):
        Lg = len(gii)
        win_range = np.arange(-(Lg // 2) + tpii, Lg - (Lg // 2) + tpii, dtype=int)
        win_range %= nn

        #        Lg2 = Lg//2
        #        oh = tpii
        #        o = oh-Lg2
        #        oe = oh+Lg2
        #
        #        if o < 0:
        #            # wraparound is in first half
        #            win_range = ((slice(o+nn,nn),slice(0,oh)),(slice(oh,oe),slice(0,0)))
        #        elif oe > nn:
        #            # wraparound is in second half
        #            win_range = ((slice(o,oh),slice(0,0)),(slice(oh,nn),slice(0,oe-nn)))
        #        else:
        #            # no wraparound
        #            win_range = ((slice(o,oh),slice(0,0)),(slice(oh,oe),slice(0,0)))

        wins.append(win_range)

    return wins, nn


#Todo : seems to have an extra window (0 index window)
if __name__ == "__main__":
    sr =44100
    L = 4096
    V = 1
    g = get_carrier_frequency_gain(V, sr)
    windows, shifts, window_widths, cf = adeliErbletWin(V, sr, L)
    window_positions,_ = calcErbWinRange(windows, shifts, L)
    plot_windows(windows,window_positions, L, sr, close = False, show= True)


