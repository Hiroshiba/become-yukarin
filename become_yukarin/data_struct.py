from typing import NamedTuple, Dict, List

import numpy
import pyworld

_min_mc = -18.3


class Wave(NamedTuple):
    wave: numpy.ndarray
    sampling_rate: int


class AcousticFeature(NamedTuple):
    f0: numpy.ndarray = numpy.nan
    spectrogram: numpy.ndarray = numpy.nan
    aperiodicity: numpy.ndarray = numpy.nan
    mfcc: numpy.ndarray = numpy.nan
    voiced: numpy.ndarray = numpy.nan

    @staticmethod
    def dtypes():
        return dict(
            f0=numpy.float32,
            spectrogram=numpy.float32,
            aperiodicity=numpy.float32,
            mfcc=numpy.float32,
            voiced=numpy.bool,
        )

    def astype(self, dtype):
        return AcousticFeature(
            f0=self.f0.astype(dtype),
            spectrogram=self.spectrogram.astype(dtype),
            aperiodicity=self.aperiodicity.astype(dtype),
            mfcc=self.mfcc.astype(dtype),
            voiced=self.voiced.astype(dtype),
        )

    def astype_only_float(self, dtype):
        return AcousticFeature(
            f0=self.f0.astype(dtype),
            spectrogram=self.spectrogram.astype(dtype),
            aperiodicity=self.aperiodicity.astype(dtype),
            mfcc=self.mfcc.astype(dtype),
            voiced=self.voiced,
        )

    def validate(self):
        assert self.f0.ndim == 2
        assert self.spectrogram.ndim == 2
        assert self.aperiodicity.ndim == 2
        assert self.mfcc.ndim == 2
        assert self.voiced.ndim == 2

        len_time = len(self.f0)
        assert len(self.spectrogram) == len_time
        assert len(self.aperiodicity) == len_time
        assert len(self.mfcc) == len_time
        assert len(self.voiced) == len_time

        assert self.voiced.dtype == numpy.bool

    @staticmethod
    def silent(length: int, sizes: Dict[str, int], keys: List[str]):
        d = {}
        if 'f0' in keys:
            d['f0'] = numpy.zeros((length, sizes['f0']), dtype=AcousticFeature.dtypes()['f0'])
        if 'spectrogram' in keys:
            d['spectrogram'] = numpy.zeros((length, sizes['spectrogram']),
                                           dtype=AcousticFeature.dtypes()['spectrogram'])
        if 'aperiodicity' in keys:
            d['aperiodicity'] = numpy.zeros((length, sizes['aperiodicity']),
                                            dtype=AcousticFeature.dtypes()['aperiodicity'])
        if 'mfcc' in keys:
            d['mfcc'] = numpy.hstack((
                numpy.ones((length, 1), dtype=AcousticFeature.dtypes()['mfcc']) * _min_mc,
                numpy.zeros((length, sizes['mfcc'] - 1), dtype=AcousticFeature.dtypes()['mfcc'])
            ))
        if 'voiced' in keys:
            d['voiced'] = numpy.zeros((length, sizes['voiced']), dtype=AcousticFeature.dtypes()['voiced'])
        feature = AcousticFeature(**d)
        return feature

    @staticmethod
    def concatenate(fs: List['AcousticFeature'], keys: List[str]):
        is_target = lambda a: not numpy.any(numpy.isnan(a))
        return AcousticFeature(**{
            key: numpy.concatenate([getattr(f, key) for f in fs]) if is_target(getattr(fs[0], key)) else numpy.nan
            for key in keys
        })

    def pick(self, first: int, last: int):
        is_target = lambda a: not numpy.any(numpy.isnan(a))
        return AcousticFeature(
            f0=self.f0[first:last] if is_target(self.f0) else numpy.nan,
            spectrogram=self.spectrogram[first:last] if is_target(self.spectrogram) else numpy.nan,
            aperiodicity=self.aperiodicity[first:last] if is_target(self.aperiodicity) else numpy.nan,
            mfcc=self.mfcc[first:last] if is_target(self.mfcc) else numpy.nan,
            voiced=self.voiced[first:last] if is_target(self.voiced) else numpy.nan,
        )

    @staticmethod
    def get_sizes(sampling_rate: int, order: int):
        fft_size = pyworld.get_cheaptrick_fft_size(fs=sampling_rate)
        return dict(
            f0=1,
            spectrogram=fft_size // 2 + 1,
            aperiodicity=fft_size // 2 + 1,
            mfcc=order + 1,
            voiced=1,
        )


class LowHighSpectrogramFeature(NamedTuple):
    low: numpy.ndarray
    high: numpy.ndarray

    def validate(self):
        assert self.low.ndim == 2
        assert self.high.ndim == 2
        assert self.low.shape == self.high.shape
