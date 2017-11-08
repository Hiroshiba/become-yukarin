import typing
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List

import chainer
import librosa
import numpy
import pysptk
import pyworld

from ..config import DatasetConfig
from ..data_struct import AcousticFeature
from ..data_struct import Wave


class BaseDataProcess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data, test):
        pass


class LambdaProcess(BaseDataProcess):
    def __init__(self, process: Callable[[any, bool], any]):
        self._process = process

    def __call__(self, data, test):
        return self._process(data, test)


class DictKeyReplaceProcess(BaseDataProcess):
    def __init__(self, key_map: Dict[str, str]):
        self._key_map = key_map

    def __call__(self, data: Dict[str, any], test):
        return {key_after: data[key_before] for key_after, key_before in self._key_map}


class ChainProcess(BaseDataProcess):
    def __init__(self, process: typing.Iterable[BaseDataProcess]):
        self._process = process

    def __call__(self, data, test):
        for p in self._process:
            data = p(data, test)
        return data


class SplitProcess(BaseDataProcess):
    def __init__(self, process: typing.Dict[str, typing.Optional[BaseDataProcess]]):
        self._process = process

    def __call__(self, data, test):
        data = {
            k: p(data, test) if p is not None else data
            for k, p in self._process.items()
        }
        return data


class WaveFileLoadProcess(BaseDataProcess):
    def __init__(self, sample_rate: int, top_db: float, dtype=numpy.float32):
        self._sample_rate = sample_rate
        self._top_db = top_db
        self._dtype = dtype

    def __call__(self, data: str, test):
        wave = librosa.core.load(data, sr=self._sample_rate, dtype=self._dtype)[0]
        wave = librosa.effects.remix(wave, intervals=librosa.effects.split(wave, top_db=self._top_db))
        return Wave(wave, self._sample_rate)


class AcousticFeatureProcess(BaseDataProcess):
    def __init__(self, frame_period, order, alpha, dtype=numpy.float32):
        self._frame_period = frame_period
        self._order = order
        self._alpha = alpha
        self._dtype = dtype

    def __call__(self, data: Wave, test):
        x = data.wave.astype(numpy.float64)
        fs = data.sampling_rate

        _f0, t = pyworld.dio(x, fs, frame_period=self._frame_period)
        f0 = pyworld.stonemask(x, _f0, t, fs)
        spectrogram = pyworld.cheaptrick(x, f0, t, fs)
        aperiodicity = pyworld.d4c(x, f0, t, fs)
        mfcc = pysptk.sp2mc(spectrogram, order=self._order, alpha=self._alpha)
        return AcousticFeature(
            f0=f0.astype(self._dtype),
            spectrogram=spectrogram.astype(self._dtype),
            aperiodicity=aperiodicity.astype(self._dtype),
            mfcc=mfcc.astype(self._dtype),
        )


class AcousticFeatureLoadProcess(BaseDataProcess):
    def __init__(self):
        pass

    def __call__(self, path: Path, test):
        d = numpy.load(path).item()  # type: dict
        return AcousticFeature(
            f0=d['f0'],
            spectrogram=d['spectrogram'],
            aperiodicity=d['aperiodicity'],
            mfcc=d['mfcc'],
        )


class AcousticFeatureNormalizeProcess(BaseDataProcess):
    def __init__(self, mean: AcousticFeature, var: AcousticFeature):
        self._mean = mean
        self._var = var

    def __call__(self, data: AcousticFeature, test):
        return AcousticFeature(
            f0=(data.f0 - self._mean.f0) / numpy.sqrt(self._var.f0),
            spectrogram=(data.spectrogram - self._mean.spectrogram) / numpy.sqrt(self._var.spectrogram),
            aperiodicity=(data.aperiodicity - self._mean.aperiodicity) / numpy.sqrt(self._var.aperiodicity),
            mfcc=(data.mfcc - self._mean.mfcc) / numpy.sqrt(self._var.mfcc),
        )


class AcousticFeatureDenormalizeProcess(BaseDataProcess):
    def __init__(self, mean: AcousticFeature, var: AcousticFeature):
        self._mean = mean
        self._var = var

    def __call__(self, data: AcousticFeature, test):
        return AcousticFeature(
            f0=data.f0 * numpy.sqrt(self._var.f0) + self._mean.f0,
            spectrogram=data.spectrogram * numpy.sqrt(self._var.spectrogram) + self._mean.spectrogram,
            aperiodicity=data.aperiodicity * numpy.sqrt(self._var.aperiodicity) + self._mean.aperiodicity,
            mfcc=data.mfcc * numpy.sqrt(self._var.mfcc) + self._mean.mfcc,
        )


class EncodeFeatureProcess(BaseDataProcess):
    def __init__(self, targets: List[str]):
        self._targets = targets

    def __call__(self, data: AcousticFeature, test):
        feature = numpy.concatenate([getattr(data, t) for t in self._targets])
        feature = feature.T
        return feature


class DecodeFeatureProcess(BaseDataProcess):
    def __init__(self, targets: List[str]):
        self._targets = targets

    def __call__(self, data: numpy.ndarray, test):
        # TODO: implement for other features
        data = data.T
        return AcousticFeature(
            f0=numpy.nan,
            spectrogram=numpy.nan,
            aperiodicity=numpy.nan,
            mfcc=data,
        )


class ShapeAlignProcess(BaseDataProcess):
    def __call__(self, data, test):
        data1, data2 = data['input'], data['target']
        m = max(data1.shape[1], data2.shape[1])
        data1 = numpy.pad(data1, ((0, 0), (0, m - data1.shape[1])), mode='constant')
        data2 = numpy.pad(data2, ((0, 0), (0, m - data2.shape[1])), mode='constant')
        data['input'], data['target'] = data1, data2
        return data


class DataProcessDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data: typing.List, data_process: BaseDataProcess):
        self._data = data
        self._data_process = data_process

    def __len__(self):
        return len(self._data)

    def get_example(self, i):
        return self._data_process(data=self._data[i], test=not chainer.config.train)


def create(config: DatasetConfig):
    import glob
    input_paths = list(sorted([Path(p) for p in glob.glob(config.input_glob)]))
    target_paths = list(sorted([Path(p) for p in glob.glob(config.target_glob)]))
    assert len(input_paths) == len(target_paths)

    acoustic_feature_load_process = AcousticFeatureLoadProcess()
    input_mean = acoustic_feature_load_process(config.input_mean_path, test=True)
    input_var = acoustic_feature_load_process(config.input_var_path, test=True)
    target_mean = acoustic_feature_load_process(config.target_mean_path, test=True)
    target_var = acoustic_feature_load_process(config.target_var_path, test=True)

    # {input_path, target_path}
    data_process = ChainProcess([
        SplitProcess(dict(
            input=ChainProcess([
                LambdaProcess(lambda d, test: d['input_path']),
                acoustic_feature_load_process,
                AcousticFeatureNormalizeProcess(mean=input_mean, var=input_var),
                EncodeFeatureProcess(['mfcc']),
            ]),
            target=ChainProcess([
                LambdaProcess(lambda d, test: d['target_path']),
                acoustic_feature_load_process,
                AcousticFeatureNormalizeProcess(mean=target_mean, var=target_var),
                EncodeFeatureProcess(['mfcc']),
            ]),
        )),
        ShapeAlignProcess(),
    ])

    num_test = config.num_test
    pairs = [
        dict(input_path=input_path, target_path=target_path)
        for input_path, target_path in zip(input_paths, target_paths)
    ]
    numpy.random.RandomState(config.seed).shuffle(pairs)
    train_paths = pairs[num_test:]
    test_paths = pairs[:num_test]
    train_for_evaluate_paths = train_paths[:num_test]

    return {
        'train': DataProcessDataset(train_paths, data_process),
        'test': DataProcessDataset(test_paths, data_process),
        'train_eval': DataProcessDataset(train_for_evaluate_paths, data_process),
    }
