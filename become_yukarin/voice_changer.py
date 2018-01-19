import numpy

from .acoustic_converter import AcousticConverter
from .super_resolution import SuperResolution


class VoiceChanger(object):
    def __init__(
            self,
            acoustic_converter: AcousticConverter,
            super_resolution: SuperResolution,
            output_sampling_rate: int = None,
    ):
        if output_sampling_rate is None:
            output_sampling_rate = super_resolution.config.dataset.param.voice_param.sample_rate

        self.acoustic_converter = acoustic_converter
        self.super_resolution = super_resolution
        self.output_sampling_rate = output_sampling_rate

    def convert_from_wave_path(self, wave_path: str):
        w_in = self.acoustic_converter._wave_process(wave_path)
        f_in = self.acoustic_converter._feature_process(w_in)
        f_low = self.acoustic_converter.convert_to_feature(f_in)
        s_high = self.super_resolution.convert(f_low.spectrogram.astype(numpy.float32))
        wave = self.super_resolution(s_high, acoustic_feature=f_low, sampling_rate=self.output_sampling_rate)
        return wave
