import fastdtw
import nnmnkwii.metrics
import numpy
import scipy.interpolate


class DTWAligner(object):
    """
    from https://github.com/r9y9/nnmnkwii/blob/4cade86b5c35b4e35615a2a8162ddc638018af0e/nnmnkwii/preprocessing/alignment.py#L14
    """

    def __init__(self, x, y, dist=lambda x, y: numpy.linalg.norm(x - y), radius=1):
        assert x.ndim == 2 and y.ndim == 2

        _, path = fastdtw.fastdtw(x, y, radius=radius, dist=dist)
        self.normed_path_x = numpy.array(list(map(lambda l: l[0], path))) / len(x)
        self.normed_path_y = numpy.array(list(map(lambda l: l[1], path))) / len(y)

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
        base = numpy.linspace(0, 1, len(normed_path))
        target = numpy.linspace(0, 1, target_length)
        path = scipy.interpolate.interp1d(base, normed_path)(target)
        path = numpy.floor(path * target_length).astype(numpy.int)
        return path


class MFCCAligner(DTWAligner):
    def __init__(self, x, y, *args, **kwargs):
        x = self._calc_aligner_feature(x)
        y = self._calc_aligner_feature(y)
        super().__init__(x, y, *args, dist=nnmnkwii.metrics.melcd, **kwargs)

    @classmethod
    def _calc_delta(cls, x):
        x = numpy.zeros_like(x, x.dtype)
        x[:-1] = x[:-1] - x[1:]
        x[-1] = 0
        return x

    @classmethod
    def _calc_aligner_feature(cls, x):
        d = cls._calc_delta(x)
        feature = numpy.concatenate((x, d), axis=1)[:, 1:]
        return feature
