import numpy as np
from filterbank.nsgt import NSGT_ERB, upsample_channels
from filterbank.erbletwin import get_carrier_frequency_gain
from filterbank.filterbank_operations import get_lateral_inhibition_coefficients, apply_inhibition, half_wave_rectify, \
    remove_DC_channels, extract_envelope, compress_coefficients, downsample, get_spectral_envelope, rms, get_roughness
from utils.plotting import plot_windows, plot_individual_channel

class TwoStageFilterBank():
    def __init__(self, first_stage_downsample = 10, second_stage_downsample=1,
                 v_first_stage=1, v_second_stage=1):
        self.v_first_stage =v_first_stage
        self.v_second_stage =v_second_stage
        self.first_stage_ds = first_stage_downsample
        self.second_stage_ds = second_stage_downsample

        self.frequency_mod_weights = [0, .8, 1, 1, .985, .970, .955, .905, .855, .805,
                                      .755, .705, .529, .353, .177, 0, 0, 0,
                                      0, 0, 0, 0, 0,0]

        self.signal_length = None
        self.sample_rate = None

    def generateFilterbanks(self, signal_length, sample_rate, ):
        self.signal_length  = signal_length
        self.sample_rate = sample_rate
        self.second_stage_sr = int(np.ceil(float(sample_rate) /
                                       self.first_stage_ds))
        self.second_stage_ls =  int(np.ceil(float(signal_length) /
                                        self.first_stage_ds))

        self.first_stage = NSGT_ERB(sample_rate, signal_length,
                                    self.v_first_stage, plot=True)
        self.second_stage = NSGT_ERB(self.second_stage_sr,
                                     self.second_stage_ls,
                                     self.v_second_stage,
                                     cutoff_frequency=200, plot=True)

        self.first_stage_center_frequencies =self.first_stage.center_frequencies
        self.second_stage_center_frequencies =self.second_stage.center_frequencies
        self.carrier_frequency_gain =  get_carrier_frequency_gain(
            self.v_first_stage, sample_rate)

    def plotFilterbank(self, stage, ipy=False, plot_responses =False):
        axes = []
        if stage == 1:
            f,ax = plot_windows(*self.first_stage.get_windows(), close= ipy)
            axes = ax
            if plot_responses:
                impulse_response = self.first_stage.get_impulse_response()
                axes += plot_individual_channel(impulse_response, 0,
                                                     len(impulse_response), close=ipy)
        if stage == 2:
            f,ax = plot_windows(*self.second_stage.get_windows(), close=ipy)
            axes = ax
            if plot_responses:

                impulse_response = self.second_stage.get_impulse_response()
                axes += plot_individual_channel(impulse_response, 0,
                                                len(impulse_response), close=ipy)
        return axes

    def forward(self, signal):
        self.raw_envelopes, self.inhibited_envelopes, self.spectral_envelope\
            =  self.applyFirstStage(signal)
        self.amplitude_modulation_envelopes, self.temporal_envelope, \
        self.amp_mod, self.mod_depth = self.applySecondStage(
            self.raw_envelopes)
        self.effective_roughness, self.instantaneous_roughness = get_roughness(
            self.amp_mod, self.mod_depth, self.second_stage_sr)

    def forward_first_stage(self, signal):
        self.raw_envelopes, self.inhibited_envelopes, self.spectral_envelope \
            = self.applyFirstStage(signal)

    def getFeatures(self):
        return self.raw_envelopes,self.inhibited_envelopes, self.spectral_envelope, self.temporal_envelope, self.amplitude_modulation_envelopes

    def applyFirstStage(self, signal):
        # Stage 1
        wins = self.first_stage.get_frequency_windows(plot=False)
        inhibitionCoefs = get_lateral_inhibition_coefficients(wins, plot=False)

        c = self.first_stage.forward(signal)
        # Extract envelopes
        envelopes = extract_envelope(c)

        # envelopes = downsample(envelopes, self.first_stage_ds)
        envelopes = upsample_channels(envelopes, self.second_stage_ls)
        envelopes = compress_coefficients(envelopes)

        # Apply lateral inhibition and rectification
        inhibitedEnvelopes = apply_inhibition(envelopes, inhibitionCoefs)
        rectified = half_wave_rectify(inhibitedEnvelopes)
        noDCChannels = remove_DC_channels(rectified)
        spectral_env = get_spectral_envelope(noDCChannels)
        return envelopes, rectified, spectral_env

    def applySecondStage(self, envelopes):
        indexes = np.arange(len(envelopes))
        epsilon = 0.0001
        #Mask inactive envelopes
        rms_values = np.zeros(envelopes.shape[0], dtype=np.float32)
        for band_index, channel in enumerate(envelopes):
            rms_values[band_index] = rms(channel)
        rms_values = rms_values / (rms_values.max()+epsilon)
        active_indexes = rms_values > 0.02

        # disregard first 2 DC channels
        active_indexes[0] = False
        active_indexes[1] = False
        active_envelope_indexes = indexes[active_indexes]

        signal_length = len(envelopes[0])
        lenvs = np.zeros((envelopes.shape[0], signal_length), dtype=np.float32)
        amp_mod = np.zeros((envelopes.shape[0], signal_length),
                           dtype=np.float32)
        modulation_depth = np.zeros((envelopes.shape[0], signal_length),
                                    dtype = np.float32)
        # First channel of filterbank used to compute modulation depth
        first_mod_ch = 1
        ds_length = signal_length//self.second_stage_ds
        if signal_length%self.second_stage_ds !=0:
            ds_length+=1

        env_shape = (envelopes.shape[0],
                     len(self.second_stage_center_frequencies), ds_length)
        stage_2_envelopes = np.zeros(env_shape, dtype = np.float32)
        for band_index in active_envelope_indexes:
            out = np.array(self.second_stage.forward_full_temp(envelopes[band_index]))

            # Last channel of filterbank used to compute modulation depth
            # Should be the last channel within the range of 1000 Hz
            # This clipping is for the roughness calculation
            # Todo : Get index of erblett at 1000Hz range from filterbank
            # for now we'll just keep all of them
            last_mod_ch = out.shape[0]

            stage_2_env = extract_envelope(out) #find envelopes of AMs to find Modulation Depths
            stage_2_envelopes_ds = downsample(stage_2_env, self.second_stage_ds)
            stage_2_envelopes[band_index] = stage_2_envelopes_ds

            # Save DC channel for later
            lenvs[band_index] = out[0]
            amp_mod[band_index] = np.sum(out[first_mod_ch: last_mod_ch, :], axis=0)

            # Calculate the parameters for roughness estimation
            d = np.diff(amp_mod[band_index])
            positive_slope_timesteps_ratio = len(d[d >= 0]) / len(d)
            shape_factor = np.exp(-(positive_slope_timesteps_ratio - 0.5) / 2)
            fmod_weights = np.reshape( self.frequency_mod_weights[
                           first_mod_ch:last_mod_ch], (1,-1))
            coefs_ac = np.matmul(fmod_weights, stage_2_env[first_mod_ch:
            last_mod_ch])
            ac_envelope = np.sqrt(np.square(coefs_ac)) #envelope of Ac component used to compute Mod depth
            max_dc = lenvs[band_index].max()
            dc_normalized = lenvs[band_index] / max_dc
            coefs_normalized=ac_envelope/max_dc
            #divide each channel by the normalized DC component to get the mod depth in that channel
            m = coefs_normalized/ (dc_normalized + np.exp(-40 * dc_normalized))
            # EXP function is large when DC is small and negligible when DC
            # is big
            m[m>1] = 1
            modulation_depth[band_index,:] = \
                self.carrier_frequency_gain[band_index] * rms_values[band_index]* shape_factor * m

        # Temporal envelope is the sum across first stage channels of the
        # second stage DC channels
        temporal_env = np.sqrt(np.sum(np.square(lenvs), axis=0))
        return stage_2_envelopes, temporal_env, amp_mod, modulation_depth







