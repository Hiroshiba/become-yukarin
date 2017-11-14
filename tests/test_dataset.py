import unittest

import numpy
from become_yukarin.dataset import dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 24000
        self.len_time = len_time = 100
        self.fft_size = fft_size = 1024
        self.order = order = 59
        self.dummy_feature = dataset.AcousticFeature(
            f0=numpy.arange(len_time).reshape((len_time, -1)),
            spectrogram=numpy.arange(len_time * (fft_size // 2 + 1)).reshape((len_time, -1)),
            aperiodicity=numpy.arange(len_time * (fft_size // 2 + 1)).reshape((len_time, -1)),
            mfcc=numpy.arange(len_time * (order + 1)).reshape((len_time, -1)),
            voiced=(numpy.arange(len_time) % 2 == 1).reshape((len_time, -1)),
        )
        self.feature_sizes = dataset.AcousticFeature.get_sizes(
            sampling_rate=self.sample_rate,
            order=self.order,
        )

    def test_encode_decode_feature(self):
        encode_feature = dataset.EncodeFeatureProcess(['mfcc'])
        decode_feature = dataset.DecodeFeatureProcess(['mfcc'], self.feature_sizes)
        e = encode_feature(self.dummy_feature, test=True)
        d = decode_feature(e, test=True)
        self.assertTrue(numpy.all(self.dummy_feature.mfcc == d.mfcc))

    def test_encode_decode_feature2(self):
        encode_feature = dataset.EncodeFeatureProcess(['mfcc', 'f0'])
        decode_feature = dataset.DecodeFeatureProcess(['mfcc', 'f0'], self.feature_sizes)
        e = encode_feature(self.dummy_feature, test=True)
        d = decode_feature(e, test=True)
        self.assertTrue(numpy.all(self.dummy_feature.mfcc == d.mfcc))
        self.assertTrue(numpy.all(self.dummy_feature.f0 == d.f0))

    def test_encode_decode_feature3(self):
        encode_feature = dataset.EncodeFeatureProcess(['mfcc', 'f0'])
        decode_feature = dataset.DecodeFeatureProcess(['mfcc', 'f0'], self.feature_sizes)
        e = encode_feature(self.dummy_feature, test=True)
        e[0] = numpy.nan
        d = decode_feature(e, test=True)
        self.assertFalse(numpy.all(self.dummy_feature.mfcc == d.mfcc))
        self.assertTrue(numpy.all(self.dummy_feature.f0 == d.f0))


if __name__ == '__main__':
    unittest.main()
