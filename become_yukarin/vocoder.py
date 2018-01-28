import numpy
import pyworld
from world4py.native import structures, apidefinitions, utils

from become_yukarin.data_struct import AcousticFeature
from become_yukarin.data_struct import Wave
from become_yukarin.dataset.dataset import AcousticFeatureProcess
from become_yukarin.param import AcousticFeatureParam


class Vocoder(object):
    def __init__(
            self,
            acoustic_feature_param: AcousticFeatureParam,
            out_sampling_rate: int,
    ):
        self.acoustic_feature_param = acoustic_feature_param
        self.out_sampling_rate = out_sampling_rate
        self._encoder = AcousticFeatureProcess(
            frame_period=acoustic_feature_param.frame_period,
            order=acoustic_feature_param.order,
            alpha=acoustic_feature_param.alpha,
        )

    def encode(self, wave: Wave):
        return self._encoder(wave)

    def decode(
            self,
            acoustic_feature: AcousticFeature,
    ):
        acoustic_feature = acoustic_feature.astype_only_float(numpy.float64)
        out = pyworld.synthesize(
            f0=acoustic_feature.f0.ravel(),
            spectrogram=acoustic_feature.spectrogram,
            aperiodicity=acoustic_feature.aperiodicity,
            fs=self.out_sampling_rate,
            frame_period=self.acoustic_feature_param.frame_period
        )
        return Wave(out, sampling_rate=self.out_sampling_rate)


class RealtimeVocoder(Vocoder):
    def __init__(
            self,
            acoustic_feature_param: AcousticFeatureParam,
            out_sampling_rate: int,
            buffer_size: int,
            number_of_pointers: int,
    ):
        super().__init__(
            acoustic_feature_param=acoustic_feature_param,
            out_sampling_rate=out_sampling_rate,
        )

        self.buffer_size = buffer_size

        self._synthesizer = structures.WorldSynthesizer()
        apidefinitions._InitializeSynthesizer(
            self.out_sampling_rate,  # sampling rate
            self.acoustic_feature_param.frame_period,  # frame period
            pyworld.get_cheaptrick_fft_size(out_sampling_rate),  # fft size
            buffer_size,  # buffer size
            number_of_pointers,  # number of pointers
            self._synthesizer,
        )
        self._before_buffer = None  # for holding memory

    def decode(
            self,
            acoustic_feature: AcousticFeature,
    ):
        length = len(acoustic_feature.f0)
        f0_buffer = utils.cast_1d_list_to_1d_pointer(acoustic_feature.f0.flatten().tolist())
        sp_buffer = utils.cast_2d_list_to_2d_pointer(acoustic_feature.spectrogram.tolist())
        ap_buffer = utils.cast_2d_list_to_2d_pointer(acoustic_feature.aperiodicity.tolist())
        apidefinitions._AddParameters(f0_buffer, length, sp_buffer, ap_buffer, self._synthesizer)

        ys = []
        while apidefinitions._Synthesis2(self._synthesizer) != 0:
            y = numpy.array([self._synthesizer.buffer[i] for i in range(self.buffer_size)])
            ys.append(y)

        if len(ys) > 0:
            out_wave = Wave(
                wave=numpy.concatenate(ys),
                sampling_rate=self.out_sampling_rate,
            )
        else:
            out_wave = Wave(
                wave=numpy.empty(0),
                sampling_rate=self.out_sampling_rate,
            )

        self._before_buffer = (f0_buffer, sp_buffer, ap_buffer)  # for holding memory
        return out_wave

    def warm_up(self, time_length: float):
        y = numpy.zeros(int(time_length * self.out_sampling_rate))
        w = Wave(wave=y, sampling_rate=self.out_sampling_rate)
        f = self.encode(w)
        self.decode(f)

    def __del__(self):
        apidefinitions._DestroySynthesizer(self._synthesizer)
