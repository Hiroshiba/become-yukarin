import json
import os
import typing
from abc import ABCMeta, abstractmethod
from typing import NamedTuple

import nnmnkwii.preprocessing
import chainer
import librosa
import numpy
import pysptk
import pyworld


class Wave(NamedTuple):
    wave: numpy.ndarray
    sampling_rate: int


class AcousticFeature(NamedTuple):
    f0: numpy.ndarray
    spectrogram: numpy.ndarray
    aperiodicity: numpy.ndarray
    mfcc: numpy.ndarray


class BaseDataProcess(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, data, test):
        pass


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


class DataProcessDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data: typing.List, data_process: BaseDataProcess, test):
        self._data = data
        self._data_process = data_process
        self._test = test

    def __len__(self):
        return len(self._data)

    def get_example(self, i):
        return self._data_process(data=self._data[i], test=self._test)


class WaveFileLoadProcess(BaseDataProcess):
    def __init__(self, sample_rate: int, top_db: float):
        self._sample_rate = sample_rate
        self._top_db = top_db

    def __call__(self, data: str, test):
        wave = librosa.core.load(data, sr=self._sample_rate)[0]
        wave = librosa.effects.remix(wave, intervals=librosa.effects.split(wave, top_db=self._top_db))
        return Wave(wave, self._sample_rate)


class AcousticFeatureProcess(BaseDataProcess):
    def __init__(self, frame_period, order, alpha):
        self._frame_period = frame_period
        self._order = order
        self._alpha = alpha

    def __call__(self, data: Wave, test):
        x = data.wave.astype(numpy.float64)
        fs = data.sampling_rate

        _f0, t = pyworld.dio(x, fs, frame_period=self._frame_period)
        f0 = pyworld.stonemask(x, _f0, t, fs)
        spectrogram = pyworld.cheaptrick(x, f0, t, fs)
        aperiodicity = pyworld.d4c(x, f0, t, fs)
        mfcc = pysptk.sp2mc(spectrogram, order=self._order, alpha=self._alpha)
        return AcousticFeature(
            f0=f0,
            spectrogram=spectrogram,
            aperiodicity=aperiodicity,
            mfcc=mfcc,
        )


# data_process = ChainProcess([
#     SplitProcess(dict(
#         input=ChainProcess([
#             WaveFileLoadProcess(),
#             AcousticFeatureProcess(),
#         ]),
#         tareget=ChainProcess([
#             WaveFileLoadProcess(),
#             AcousticFeatureProcess(),
#         ]),
#     )),
#
#     PILImageProcess(mode='RGB'),
#     RandomFlipImageProcess(p_flip_horizontal=0.5, p_flip_vertical=0),
#     RandomResizeImageProcess(min_short=128, max_short=160),
#     RandomCropImageProcess(crop_width=128, crop_height=128),
#     RgbImageArrayProcess(),
#     SplitProcess({
#         'target': None,
#         'raw_line': RawLineImageArrayProcess(),
#     })
# ])
#
#
# def choose(config: DatasetConfig):
#     if config.images_glob is not None:
#         import glob
#         paths = glob.glob(config.images_glob)
#         paths = data_filter(
#             datas=paths,
#             keys=list(map(lambda p: os.path.basename(p), paths)),
#             filter_func=filter_image,
#             num_process=None,
#             cache_path=config.cache_path,
#         )
#         paths = list(paths)
#     else:
#         paths = json.load(open(config.images_list))
#
#     num_test = config.num_test
#     train_paths = paths[num_test:]
#     test_paths = paths[:num_test]
#     train_for_evaluate_paths = train_paths[:num_test]
#
#     return {
#         'train': DataProcessDataset(train_paths, data_process, test=False),
#         'test': DataProcessDataset(test_paths, data_process, test=True),
#         'train_eval': DataProcessDataset(train_for_evaluate_paths, data_process, test=True),
#     }
