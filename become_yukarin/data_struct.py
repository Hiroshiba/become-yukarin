from typing import NamedTuple

import pyworld

import numpy


class Wave(NamedTuple):
    wave: numpy.ndarray
    sampling_rate: int


class AcousticFeature(NamedTuple):
    f0: numpy.ndarray
    spectrogram: numpy.ndarray
    aperiodicity: numpy.ndarray
    mfcc: numpy.ndarray
    voiced: numpy.ndarray

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
    def get_sizes(sampling_rate: int, order: int):
        fft_size = pyworld.get_cheaptrick_fft_size(fs=sampling_rate)
        return dict(
            f0=1,
            spectrogram=fft_size // 2 + 1,
            aperiodicity=fft_size // 2 + 1,
            mfcc=order + 1,
            voiced=1,
        )
