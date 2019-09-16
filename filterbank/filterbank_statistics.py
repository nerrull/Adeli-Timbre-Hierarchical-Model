import numpy as np
from utils.windows import tukey_win
from matplotlib import pyplot as plt

def make_unit_window(l):
    transition = min(512, l / 4)
    window = tukey_win(l, tr_area=transition)
    return window / np.sum(window)


def marginal_statistics(envelope, epsilon =1e-6):
    window = make_unit_window(envelope.shape[1]).reshape(1,-1).repeat(
        envelope.shape[0], axis=0)
    #Mean
    M1 = np.sum(window*envelope, axis =1)

    centered_envelopes = (envelope.T -M1).T
    centered_envelopes[centered_envelopes<1e-2]=0

    centered_envelopes_2 = np.square(centered_envelopes)
    centered_envelopes_3 =centered_envelopes_2 * centered_envelopes
    centered_envelopes_4 =centered_envelopes_3 * centered_envelopes

    #Normalized Variance
    variance = np.sum(window *centered_envelopes_2, axis=1 )
    std_dev = np.sqrt(variance)

    M2 = variance/(M1+epsilon)

    #Skew
    M3 = np.sum(window*centered_envelopes_3, axis=1)/((std_dev+epsilon)**3)
    #Kurtosis
    M4 = np.sum(window*centered_envelopes_4, axis=1)/((std_dev+epsilon)**4)

    # if standard deviation is very small, we essentially have a dirace
    # distribution so set the higher order moments to zero (otherwise they
    # blow up because variance is so small
    M3[variance <1e-2] = 0
    M4[variance <1e-2] = 0

    return M1, M2,M3, M4, variance, std_dev


def subband_cross_correlations(envelopes, means, variances):
    window = make_unit_window(envelopes.shape[1])

    num_envelopes = len(envelopes)
    envs = envelopes -means.reshape(-1,1)
    std_devs=  np.sqrt(variances)
    corr = np.zeros((num_envelopes, num_envelopes))
    for j,(s_j,sigma_j) in enumerate(zip(envs, std_devs)):
        for k, (s_k, sigma_k) in list(enumerate(zip(envs, std_devs)))[j+1:]:
            corr[j,k] = np.sum(window*s_j*s_k/(sigma_j*sigma_k))
    return corr

def subband_cross_correlations_mcdermott(envelopes, means, variances):
    steps = [1,2,3,4,5,6,11,16,21]
    window = make_unit_window(envelopes.shape[1])

    num_envelopes = len(envelopes)
    envs = envelopes -means.reshape(-1,1)
    std_devs=  np.sqrt(variances)
    corr = np.zeros((num_envelopes, len(steps)))
    unwrapped_corr = np.array([])

    for j,(s_j,sigma_j) in enumerate(zip(envs, std_devs)):
        corrs = []
        for i, k_j in enumerate(steps):
            if k_j + j >=num_envelopes:
                break
            k = j +k_j
            s_k = envs[k]
            sigma_k = std_devs[k]
            if sigma_j==0 or sigma_k ==0: corr [ j,i] = 0.
            else: corr[j,i] = np.sum(window*s_j*s_k/(sigma_j*sigma_k))
            corrs.append(corr[j,i])
        unwrapped_corr = np.append( unwrapped_corr, np.array(corrs))
    return corr, unwrapped_corr

def modulation_powers(bands_envelopes, variances):
    num_bands, num_envelopes = bands_envelopes.shape[0:2]
    mod_powers = np.zeros((num_bands, num_envelopes), dtype = np.float32)
    for i, (band, variance) in enumerate(zip(bands_envelopes, variances)):
        for j, envelope in enumerate(band):
            if variance <1e-3:
                mod_powers[i, j] =0
            else:
                mod_powers[i,j] = modulation_power(envelope, variance)
    return mod_powers


def modulation_power(envelope, variance):
    window = make_unit_window(envelope.shape[0])
    return np.sum(window*np.square(envelope)/variance)

def max_trigger_autocorrelation(envelope, timestep):
    #Cut off the end that doesnt fit nicely with the timestep
    if (len(envelope)%timestep) !=0:
        envelope = envelope[: - (len(envelope)%timestep)]
    envelope_segments = envelope.reshape((-1,timestep))
    #Get max indexes
    segment_maxes =  np.argmax(envelope_segments, axis =1)
    segment_max_times = np.array([segment_max + timestep*i for i, segment_max in enumerate(segment_maxes)])
    aligned_segments = np.zeros((len(segment_maxes),timestep*2))

    # Pad the envelope for start and end frames to line up properly
    envelope= np.pad(envelope, timestep, "constant", constant_values=0)
    #Offset the times to compensate for the padding
    segment_max_times = segment_max_times + timestep

    #Line them up
    for i,trigger_time in enumerate(segment_max_times):
        aligned_segments[i, :] = envelope[trigger_time - timestep: trigger_time + timestep]

    reduced = aligned_segments.sum( axis =0)
    return reduced

