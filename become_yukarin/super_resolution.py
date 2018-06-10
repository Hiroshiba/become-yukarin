from functools import partial
from pathlib import Path
from typing import List

import chainer
import numpy
import pyworld

from become_yukarin.config.sr_config import SRConfig
from become_yukarin.data_struct import AcousticFeature
from become_yukarin.data_struct import Wave
from become_yukarin.dataset.dataset import LowHighSpectrogramFeatureLoadProcess
from become_yukarin.dataset.dataset import LowHighSpectrogramFeatureProcess
from become_yukarin.dataset.dataset import WaveFileLoadProcess
from become_yukarin.model.sr_model import create_predictor_sr


class SuperResolution(object):
    def __init__(self, config: SRConfig, model_path: Path, gpu: int = None) -> None:
        self.config = config
        self.model_path = model_path
        self.gpu = gpu

        self.model = model = create_predictor_sr(config.model)
        chainer.serializers.load_npz(str(model_path), model)
        if self.gpu is not None:
            model.to_gpu(self.gpu)

        self._param = param = config.dataset.param
        self._wave_process = WaveFileLoadProcess(
            sample_rate=param.voice_param.sample_rate,
            top_db=None,
        )
        self._low_high_spectrogram_process = LowHighSpectrogramFeatureProcess(
            frame_period=param.acoustic_feature_param.frame_period,
            order=param.acoustic_feature_param.order,
            alpha=param.acoustic_feature_param.alpha,
            f0_estimating_method=param.acoustic_feature_param.f0_estimating_method,
        )
        self._low_high_spectrogram_load_process = LowHighSpectrogramFeatureLoadProcess(
            validate=True,
        )

    def convert(self, input: numpy.ndarray) -> numpy.ndarray:
        converter = partial(chainer.dataset.convert.concat_examples, device=self.gpu, padding=0)
        pad = 128 - len(input) % 128
        input = numpy.pad(input, [(0, pad), (0, 0)], mode='minimum')
        input = numpy.log(input)[:, :-1]
        input = input[numpy.newaxis]
        inputs = converter([input])

        with chainer.using_config('train', False):
            out = self.model(inputs).data[0]

        if self.gpu is not None:
            out = chainer.cuda.to_cpu(out)

        out = out[0]
        out = numpy.pad(out, [(0, 0), (0, 1)], mode='edge')
        out = numpy.exp(out)
        out = out[:-pad]
        return out

    def convert_loop(self, input: numpy.ndarray, n_len: int = 512, n_wrap: int = 128):
        out_feature_list: List[AcousticFeature] = []
        N = len(input)
        for i in numpy.arange(0, int(numpy.ceil(N / n_len))):
            # convert with overwrapped
            start = i * n_len
            mi = max(start - n_wrap, 0)
            ma = min(start + n_len + n_wrap, N)
            f = input[numpy.arange(mi, ma)]
            o_warp = self.convert(f)

            # eliminate overwrap
            ex_mi = start - mi
            ex_len = min(ma - start, n_len)
            o = o_warp[numpy.arange(ex_mi, ex_mi + ex_len)]
            out_feature_list.append(o)
        return numpy.concatenate(out_feature_list)

    def convert_to_feature(
            self,
            spectrogram: numpy.ndarray,
            acoustic_feature: AcousticFeature,
    ):
        acoustic_feature = acoustic_feature.astype_only_float(numpy.float64)
        f_out = AcousticFeature(
            f0=acoustic_feature.f0,
            spectrogram=spectrogram.astype(numpy.float64),
            aperiodicity=acoustic_feature.aperiodicity,
            mfcc=acoustic_feature.mfcc,
            voiced=acoustic_feature.voiced,
        )
        return f_out

    def convert_to_audio(
            self,
            input: numpy.ndarray,
            acoustic_feature: AcousticFeature,
            sampling_rate: int,
    ):
        acoustic_feature = acoustic_feature.astype_only_float(numpy.float64)
        out = pyworld.synthesize(
            f0=acoustic_feature.f0.ravel(),
            spectrogram=input.astype(numpy.float64),
            aperiodicity=acoustic_feature.aperiodicity,
            fs=sampling_rate,
            frame_period=self._param.acoustic_feature_param.frame_period,
        )
        return Wave(out, sampling_rate=sampling_rate)

    def convert_from_audio_path(self, input: Path):
        wave = self._wave_process(str(input), test=True)
        feature = self._low_high_spectrogram_process(wave, test=True)
        return self.convert(feature.low)

    def convert_from_feature_path(self, input: Path):
        feature = self._low_high_spectrogram_load_process(input, test=True)
        return self.convert(feature.low)

    def __call__(
            self,
            input: numpy.ndarray,
            acoustic_feature: AcousticFeature,
            sampling_rate: int,
    ):
        high = self.convert(input)
        return self.convert_to_audio(high, acoustic_feature=acoustic_feature, sampling_rate=sampling_rate)
