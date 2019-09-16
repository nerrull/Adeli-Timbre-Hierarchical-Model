import matplotlib.pyplot as plt

def stage_one_analyis(filterbank, plot = True):
    inhibited_envelopes = filterbank.inhibited_envelopes

    mean_inhib, norm_var_inhib, skew_inhib, kurtosis_inhib, variance_inhib,_ =\
        marginal_statistics(inhibited_envelopes)

    cross_corr_inhib, unwrapped = subband_cross_correlations_mcdermott(
        inhibited_envelopes,
                                                   mean_inhib, variance_inhib)

    envelopes = filterbank.raw_envelopes

    mean, norm_var, skew, kurtosis, variance, _ = marginal_statistics(
        envelopes)

    cross_corr, unwrapped = \
        subband_cross_correlations_mcdermott(envelopes,mean,variance)

    if plot:
        plot_marginal("Spectral envelope moments",mean, norm_var, skew,
                      kurtosis)
        plot_marginal("Inhibited spectral envelope moments",mean_inhib,
                      norm_var_inhib,
                      skew_inhib, kurtosis_inhib)

        plot_correlation(cross_corr, "Compressed env cross-correlation",
                         close=False)
        plot_correlation(cross_corr_inhib, "Inhibited env cross-correlation",
                         close=False)
        plt.show()

    return variance

if __name__ == "__main__":

    from utils.file_utils import LibrosaSoundFileLoader
    from filterbank.filterbank_statistics import  marginal_statistics,\
        subband_cross_correlations,modulation_powers,\
        subband_cross_correlations_mcdermott
    from utils.plotting import plot_correlation, plot_marginal, \
        plot_modulation_power
    from utils.windows import tukey_win

    from filterbank.two_stage_filterbank import TwoStageFilterBank
    filename = '../data/concat1.wav'
    filename = '../data/bass_synthetic_046_pack.wav'
    #filename = '../data/dayvan.wav'


    num_seconds = 10
    segment_offset_seconds = 0

    # Load the file
    sf = LibrosaSoundFileLoader(filename)
    signal = sf.get_segment(num_seconds, segment_offset_seconds)
    win = tukey_win(len(signal), 1500)
    signal = signal*win

    filterbank = TwoStageFilterBank(v_first_stage=1)
    filterbank.generateFilterbanks(len(signal), sample_rate=sf.get_sample_rate())
    # axes = filterbank.plotFilterbank(1, ipy=False)
    filterbank.forward(signal)

    raw_envelopes = filterbank.raw_envelopes
    inhibited_envelopes = filterbank.inhibited_envelopes

    f = plt.figure(figsize=(20, 10))
    im = plt.imshow(raw_envelopes, aspect=float(raw_envelopes.shape[1]) / raw_envelopes.shape[0] * 0.5,
                   interpolation='nearest')
    plt.title("ERB band envelopes")
    plt.xlabel("Time")
    plt.ylabel("ERB band index")


    plt.figure()
    plt.title("Instantaneous Roughness")
    plt.plot(filterbank.instantaneous_roughness)

    variance = stage_one_analyis(filterbank, plot = True)

    mp = modulation_powers(filterbank.amplitude_modulation_envelopes, variance)
    plot_modulation_power(mp,axes = [filterbank.first_stage_center_frequencies.astype(np.int32), filterbank.second_stage_center_frequencies.astype(np.int32)], close = False)

    plt.show()

