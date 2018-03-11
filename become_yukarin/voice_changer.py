import numpy

from .acoustic_converter import AcousticConverter
from .data_struct import AcousticFeature
from .super_resolution import SuperResolution


class VoiceChanger(object):
    def __init__(
            self,
            acoustic_converter: AcousticConverter,
            super_resolution: SuperResolution,
            output_sampling_rate: int = None,
    ) -> None:
        if output_sampling_rate is None:
            output_sampling_rate = super_resolution.config.dataset.param.voice_param.sample_rate

        self.acoustic_converter = acoustic_converter
        self.super_resolution = super_resolution
        self.output_sampling_rate = output_sampling_rate

    def convert_from_acoustic_feature(self, f_in: AcousticFeature):
        f_low = self.acoustic_converter.convert_to_feature(f_in)
        s_high = self.super_resolution.convert(f_low.spectrogram.astype(numpy.float32))
        f_high = self.super_resolution.convert_to_feature(s_high, f_low)
        return f_high
