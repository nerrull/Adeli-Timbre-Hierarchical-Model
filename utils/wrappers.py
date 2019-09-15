
from librosa.feature import mfcc,delta, melspectrogram
from librosa.filters import get_window
from librosa.core.spectrum import stft
from scipy.fftpack import dct
from filterbank.windows import Windower, HannWindower
from filterbank.two_stage_filterbank import TwoStageFilterBank
from filterbank.filterbank_statistics import modulation_powers, marginal_statistics
import numpy as np
import pandas as pd


class FilterbankWrapper():
    def __init__(self,  sample_rate, window_size, window_stride):
        self.sample_rate = sample_rate
        self.filterbank = TwoStageFilterBank(first_stage_downsample=10, second_stage_downsample=1, v_first_stage=1)
        self.filterbank.generateFilterbanks(window_size, sample_rate=sample_rate)
        self.windower = HannWindower(window_size,  stride=window_stride)
        self.n_bands = len(self.filterbank.first_stage.center_frequencies)

    def get_num_bands(self):
        return self.n_bands

    def process_signal(self, signal):
        spectral_envs =[]
        amplitude_envs= []
        full_env = []
        temporal_env = []
        temporal_env_full = []
        dcts = []

        for i,slice in enumerate(self.windower.forward(signal)):
            raw_env, env_inh, spectral_env, temp_env, amp_env = \
                self.filterbank.forward(slice)
            spectral_env = spectral_env.reshape((-1,1))
            amp_env = np.mean(amp_env, axis =2).reshape((self.n_bands, -1,1))
            if i==0:
                spectral_envs= spectral_env
                amplitude_envs= amp_env
                full_env =env_inh
                temporal_env = np.array(np.mean(temp_env))
                temporal_env_full = np.array(temp_env)
            else:
                spectral_envs =np.concatenate((spectral_envs,spectral_env), axis=1)
                amplitude_envs =np.concatenate((amplitude_envs,amp_env), axis=2)
                full_env = np.concatenate((full_env, env_inh), axis =1)
                temporal_env = np.append(temporal_env,np.array(np.mean(temp_env)))
                temporal_env_full = self.windower.combine(temporal_env_full, temp_env, 10)

        dcts = dct(spectral_envs, norm="ortho", axis = 0)
        dct_raw = dct(full_env, norm="ortho", axis = 0)

        return spectral_envs, amplitude_envs, full_env, temporal_env, \
               temporal_env_full, dcts, dct_raw

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self

class FullSignalFilterbankWrapper():
    def __init__(self,  sample_rate, signal_length):
        self.sample_rate = sample_rate
        self.filterbank = TwoStageFilterBank(first_stage_downsample=10, second_stage_downsample=1, v_first_stage=1)
        self.filterbank.generateFilterbanks(signal_length, sample_rate=sample_rate)
        self.n_bands = len(self.filterbank.first_stage.center_frequencies)
        self.samples_twentyms = int(sample_rate / 1000 * 20)
        self.samples_twenty_ms_ds = int(self.samples_twentyms / 10)
        #Extracted features dict
        self.efd = DotDict()
        self.dctf = DotDict()
        self.envs =DotDict()

    def get_num_bands(self):
        return self.n_bands

    def process_signal(self, signal):
        self.filterbank.forward(signal)

        self.envs.raw_env = self.filterbank.raw_envelopes
        self.envs.inh_env = self.filterbank.inhibited_envelopes
        self.envs.amp_env = self.filterbank.amplitude_modulation_envelopes
        self.envs.amp_mod = self.filterbank.amp_mod

        self.efd.spectral_env = self.filterbank.spectral_envelope
        self.efd.effective_roughness = self.filterbank.effective_roughness
        # self.efd.mod_depth = self.filterbank.mod_depth

        #Calculate marginal statistics
        self.efd.inh_stats = marginal_statistics(self.envs.inh_env)
        self.efd.raw_stats = marginal_statistics(self.envs.raw_env)

        m, v_unitless, s, k, var, std_dev = self.efd.raw_stats

        #Calculate modulation features
        self.efd.modulation_power = modulation_powers(self.envs.amp_env, var)
        self.efd.average_amp_mod = np.mean(self.envs.amp_env, axis=2).reshape((
            self.n_bands, -1))

        temp_env = self.filterbank.temporal_envelope
        inst_roughness = self.filterbank.instantaneous_roughness

        #Make temporal env resolution 60 ms
        diff = len(temp_env) % self.samples_twenty_ms_ds
        if diff !=0 :
            pad =  np.zeros(self.samples_twenty_ms_ds - diff)
            temp_env = np.append(temp_env, pad)
            inst_roughness = np.append(inst_roughness, pad)

        self.efd.temp_env_reduced = np.mean(np.reshape(
            temp_env,(-1,self.samples_twenty_ms_ds)), axis=1)
        self.envs.temp_env = temp_env

        self.efd.inst_roughness = np.mean(np.reshape(
            inst_roughness,(-1,self.samples_twenty_ms_ds)), axis=1)

        # #Also make raw env resolution 60 ms turns out this doesnt improve
        # dct speed by much at all
        # diff2 = len(raw_env[0]) %self.samples_sixtyms_ds
        # if diff2 !=0 :
        #     pad = np.zeros((self.n_bands, self.samples_sixtyms_ds-diff2))
        #     raw_env = np.hstack((raw_env,pad ))
        #     env_inh = np.hstack((env_inh, pad))
        #
        # raw_env_reduced = np.mean(np.reshape(
        #     raw_env,(self.n_bands,-1, self.samples_sixtyms_ds)), axis=2)
        # env_inh_reduced = np.mean(np.reshape(
        #     env_inh, (self.n_bands, -1, self.samples_sixtyms_ds)), axis=2)

        # dct_raw = dct(raw_env_reduced, norm="ortho", axis = 0)
        # dct_inh = dct(env_inh_reduced, norm="ortho", axis = 0)
        #
        # dct_raw = dct(raw_env, norm="ortho", axis = 0)

        #Compute dct on envelopes
        self.dctf.dct_inhibited = dct(self.envs.inh_env, norm="ortho",
                                      axis = 0)
        self.dctf.dct_delta = delta(self.dctf.dct_inhibited)
        self.dctf.dct_delta_delta = delta(self.dctf.dct_delta )

        self.efd.dct = np.mean(self.dctf.dct_inhibited, axis =1)
        self.efd.dct_delta = np.mean(self.dctf.dct_delta, axis =1)
        self.efd.dct_delta_delta = np.mean(self.dctf.dct_delta_delta, axis =1)


    def to_dataframe(self):
        return  pd.DataFrame({ key:[value] for key, value in
                               self.efd.items() })

class MFCCWrapper():
    def __init__(self, sample_rate, window_size, window_stride, num_mfccs):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.num_mfccs = num_mfccs

    def process_signal(self, signal):
        ft = np.abs(stft(signal, n_fft=self.window_size, hop_length=self.window_stride, window='hann'))
        mel = melspectrogram(sr=self.sample_rate,S=ft)
        mfccs = mfcc( sr=self.sample_rate, n_mfcc=self.num_mfccs,S=mel)
        deltas=  delta(mfccs)
        delta_deltas=  delta(mfccs,order=2)
        return mfccs, deltas, delta_deltas

