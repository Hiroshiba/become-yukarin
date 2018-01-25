from functools import partial
from pathlib import Path
from typing import Optional

import chainer
import numpy
import pysptk
import pyworld

from become_yukarin.config.config import Config
from become_yukarin.data_struct import AcousticFeature
from become_yukarin.data_struct import Wave
from become_yukarin.dataset.dataset import AcousticFeatureDenormalizeProcess
from become_yukarin.dataset.dataset import AcousticFeatureLoadProcess
from become_yukarin.dataset.dataset import AcousticFeatureNormalizeProcess
from become_yukarin.dataset.dataset import AcousticFeatureProcess
from become_yukarin.dataset.dataset import DecodeFeatureProcess
from become_yukarin.dataset.dataset import EncodeFeatureProcess
from become_yukarin.dataset.dataset import WaveFileLoadProcess
from become_yukarin.model.model import create_predictor


class AcousticConverter(object):
    def __init__(self, config: Config, model_path: Path, gpu: int = None) -> None:
        self.config = config
        self.model_path = model_path
        self.gpu = gpu

        self.model = model = create_predictor(config.model)
        chainer.serializers.load_npz(str(model_path), model)
        if self.gpu is not None:
            model.to_gpu(self.gpu)

        self._param = param = config.dataset.param
        self._wave_process = WaveFileLoadProcess(
            sample_rate=param.voice_param.sample_rate,
            top_db=None,
        )
        self._feature_process = AcousticFeatureProcess(
            frame_period=param.acoustic_feature_param.frame_period,
            order=param.acoustic_feature_param.order,
            alpha=param.acoustic_feature_param.alpha,
            f0_estimating_method=param.acoustic_feature_param.f0_estimating_method,
        )

        self._acoustic_feature_load_process = acoustic_feature_load_process = AcousticFeatureLoadProcess()

        input_mean = acoustic_feature_load_process(config.dataset.input_mean_path, test=True)
        input_var = acoustic_feature_load_process(config.dataset.input_var_path, test=True)
        target_mean = acoustic_feature_load_process(config.dataset.target_mean_path, test=True)
        target_var = acoustic_feature_load_process(config.dataset.target_var_path, test=True)
        self._feature_normalize = AcousticFeatureNormalizeProcess(
            mean=input_mean,
            var=input_var,
        )
        self._feature_denormalize = AcousticFeatureDenormalizeProcess(
            mean=target_mean,
            var=target_var,
        )

        feature_sizes = AcousticFeature.get_sizes(
            sampling_rate=param.voice_param.sample_rate,
            order=param.acoustic_feature_param.order,
        )
        self._encode_feature = EncodeFeatureProcess(config.dataset.features)
        self._decode_feature = DecodeFeatureProcess(config.dataset.features, feature_sizes)

    def convert_to_feature(self, input: AcousticFeature, out_sampling_rate: Optional[int] = None):
        if out_sampling_rate is None:
            out_sampling_rate = self.config.dataset.param.voice_param.sample_rate

        input_feature = input
        input = self._feature_normalize(input, test=True)
        input = self._encode_feature(input, test=True)

        converter = partial(chainer.dataset.convert.concat_examples, device=self.gpu, padding=0)
        inputs = converter([input])

        with chainer.using_config('train', False):
            out = self.model(inputs).data[0]

        if self.gpu is not None:
            out = chainer.cuda.to_cpu(out)

        out = self._decode_feature(out, test=True)
        out = AcousticFeature(
            f0=out.f0,
            spectrogram=out.spectrogram,
            aperiodicity=out.aperiodicity,
            mfcc=out.mfcc,
            voiced=input_feature.voiced,
        )
        out = self._feature_denormalize(out, test=True)
        out = AcousticFeature(
            f0=out.f0,
            spectrogram=out.spectrogram,
            aperiodicity=input_feature.aperiodicity,
            mfcc=out.mfcc,
            voiced=out.voiced,
        )

        fftlen = pyworld.get_cheaptrick_fft_size(out_sampling_rate)
        spectrogram = pysptk.mc2sp(
            out.mfcc,
            alpha=self._param.acoustic_feature_param.alpha,
            fftlen=fftlen,
        )

        out = AcousticFeature(
            f0=out.f0,
            spectrogram=spectrogram,
            aperiodicity=out.aperiodicity,
            mfcc=out.mfcc,
            voiced=out.voiced,
        ).astype(numpy.float64)
        return out

    def convert_from_audio_path(self, path: Path, out_sampling_rate: Optional[int] = None):
        wave = self._wave_process(str(path), test=True)
        feature = self._feature_process(wave, test=True)
        return self.convert_from_feature(feature, out_sampling_rate)

    def convert_from_feature_path(self, path: Path, out_sampling_rate: Optional[int] = None):
        feature = self._acoustic_feature_load_process(path, test=True)
        return self.convert_from_feature(feature, out_sampling_rate)

    def convert_from_feature(self, input: AcousticFeature, out_sampling_rate: Optional[int] = None):
        if out_sampling_rate is None:
            out_sampling_rate = self.config.dataset.param.voice_param.sample_rate

        out = self.convert_to_feature(input=input, out_sampling_rate=out_sampling_rate)
        out = pyworld.synthesize(
            f0=out.f0.ravel(),
            spectrogram=out.spectrogram,
            aperiodicity=out.aperiodicity,
            fs=out_sampling_rate,
            frame_period=self._param.acoustic_feature_param.frame_period,
        )
        return Wave(out, sampling_rate=out_sampling_rate)

    def __call__(self, voice_path: Path, out_sampling_rate: Optional[int] = None):
        return self.convert_from_audio_path(voice_path, out_sampling_rate)
