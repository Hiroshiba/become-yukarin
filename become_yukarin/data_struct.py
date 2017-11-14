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
            voiced=self.mfcc.astype(dtype),
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
