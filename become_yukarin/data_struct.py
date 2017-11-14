from typing import NamedTuple

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
