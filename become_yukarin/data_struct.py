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
