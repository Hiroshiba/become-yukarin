import math

import fastdtw
import numpy

_logdb_const = 10.0 / numpy.log(10.0) * numpy.sqrt(2.0)


# should work on torch and numpy arrays
def _sqrt(x):
    isnumpy = isinstance(x, numpy.ndarray)
    isscalar = numpy.isscalar(x)
    return numpy.sqrt(x) if isnumpy else math.sqrt(x) if isscalar else x.sqrt()


def _exp(x):
    isnumpy = isinstance(x, numpy.ndarray)
    isscalar = numpy.isscalar(x)
    return numpy.exp(x) if isnumpy else math.exp(x) if isscalar else x.exp()


def _sum(x):
    if isinstance(x, list) or isinstance(x, numpy.ndarray):
        return numpy.sum(x)
    return float(x.sum())


def melcd(X, Y, lengths=None):
    """Mel-cepstrum distortion (MCD).

    The function computes MCD for time-aligned mel-cepstrum sequences.

    Args:
        X (ndarray): Input mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        Y (ndarray): Target mel-cepstrum, shape can be either of
          (``D``,), (``T x D``) or (``B x T x D``). Both Numpy and torch arrays
          are supported.
        lengths (list): Lengths of padded inputs. This should only be specified
          if you give mini-batch inputs.

    Returns:
        float: Mean mel-cepstrum distortion in dB.

    .. note::

        The function doesn't check if inputs are actually mel-cepstrum.
    """
    # summing against feature axis, and then take mean against time axis
    # Eq. (1a)
    # https://www.cs.cmu.edu/~awb/papers/sltu2008/kominek_black.sltu_2008.pdf
    if lengths is None:
        z = X - Y
        r = _sqrt((z * z).sum(-1))
        if not numpy.isscalar(r):
            r = r.mean()
        return _logdb_const * r

    # Case for 1-dim features.
    if len(X.shape) == 2:
        # Add feature axis
        X, Y = X[:, :, None], Y[:, :, None]

    s = 0.0
    T = _sum(lengths)
    for x, y, length in zip(X, Y, lengths):
        x, y = x[:length], y[:length]
        z = x - y
        s += _sqrt((z * z).sum(-1)).sum()

    return _logdb_const * s / T


class DTWAligner(object):
    """
    from https://github.com/r9y9/nnmnkwii/blob/4cade86b5c35b4e35615a2a8162ddc638018af0e/nnmnkwii/preprocessing/alignment.py#L14
    """

    def __init__(self, x, y, dist=lambda x, y: numpy.linalg.norm(x - y), radius=1) -> None:
        assert x.ndim == 2 and y.ndim == 2

        _, path = fastdtw.fastdtw(x, y, radius=radius, dist=dist)
        path = numpy.array(path)
        self.normed_path_x = path[:, 0] / len(x)
        self.normed_path_y = path[:, 1] / len(y)

    def align_x(self, x):
        path = self._interp_path(self.normed_path_x, len(x))
        return x[path]

    def align_y(self, y):
        path = self._interp_path(self.normed_path_y, len(y))
        return y[path]

    def align(self, x, y):
        return self.align_x(x), self.align_y(y)

    @staticmethod
    def align_and_transform(x, y, *args, **kwargs):
        aligner = DTWAligner(*args, x=x, y=y, **kwargs)
        return aligner.align(x, y)

    @staticmethod
    def _interp_path(normed_path: numpy.ndarray, target_length: int):
        path = numpy.floor(normed_path * target_length).astype(numpy.int)
        return path


class MFCCAligner(DTWAligner):
    def __init__(self, x, y, *args, **kwargs) -> None:
        x = self._calc_aligner_feature(x)
        y = self._calc_aligner_feature(y)
        kwargs.update(dist=melcd)
        super().__init__(x, y, *args, **kwargs)

    @classmethod
    def _calc_delta(cls, x):
        x = numpy.zeros_like(x, x.dtype)
        x[:-1] = x[1:] - x[:-1]
        x[-1] = 0
        return x

    @classmethod
    def _calc_aligner_feature(cls, x):
        d = cls._calc_delta(x)
        feature = numpy.concatenate((x, d), axis=1)[:, 1:]
        return feature
