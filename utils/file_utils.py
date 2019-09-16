from pydub import AudioSegment
import librosa

import numpy as np
import wave
import struct

def export_multichannel(channels, framerate, outfile):
    channels_16 = []
    for index, channel in enumerate(channels):
        channel_int16 = (np.real(channel)).astype(np.int16)
        channel_int16 = (AudioSegment(
            channel_int16.tobytes(), frame_rate=framerate,
            sample_width=channel_int16.dtype.itemsize,
            channels=1))
        channels_16.append(channel_int16)

    output = AudioSegment.from_mono_audiosegments(*channels_16)
    output.export(out_f=outfile, format='wav')


def loadFile(filename):
    wr = wave.open(filename, mode='rb')
    return wr

class SoundFileLoader:
    def __init__(self, filename):
        self.load_file(filename)

    def load_file(self, filename):
        self.wavereader = wave.open(filename, mode='rb')
        self.sample_rate = self.wavereader.getframerate()
        self.sample_width = self.wavereader.getsampwidth()
        self.nframes = self.wavereader.getnframes()
        self.nchannels = self.wavereader.getnchannels()

    def get_segment(self, num_seconds, segment_offset_seconds =0):
        # Extract the audio segment data
        segment_offset = int(segment_offset_seconds * self.sample_rate)
        segment_length = int(num_seconds * self.sample_rate)
        discard = self.wavereader.readframes(segment_offset)
        seg = self.wavereader.readframes(segment_length)
        num_h = int(self.sample_width * segment_length / 2)
        segment = struct.unpack('h' * num_h * self.nchannels, seg)
        # get first channel
        segment = segment[0::self.nchannels]
        return segment
    def get_sample_rate(self):
        return self.sample_rate


class LibrosaSoundFileLoader:
    def __init__(self, filename):
        self.filename =filename

    def get_segment(self, num_seconds= None, segment_offset_seconds =0):
        segment, sr = librosa.core.load(self.filename, sr=None, duration=num_seconds, offset= segment_offset_seconds)
        self.sample_rate =sr
        return segment
    def get_sample_rate(self):
        return self.sample_rate

