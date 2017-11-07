import chainer

from .config import ModelConfig


class DeepConvolution1D(chainer.link.Chain):
    def __init__(self, in_size: int, num_scale: int, base_num_z: int, **kwargs):
        super().__init__(**kwargs)
        self.num_scale = num_scale
        self.out_size = base_num_z * 2 ** (num_scale - 1)

        for i in range(num_scale):
            l = base_num_z * 2 ** i
            self.add_link('conv{}'.format(i + 1), chainer.links.ConvolutionND(1, in_size, l, 3, 1, 1, nobias=True))
            self.add_link('bn{}'.format(i + 1), chainer.links.BatchNormalization(l))
            in_size = l

    def __call__(self, x):
        h = x
        for i in range(self.num_scale):
            conv = getattr(self, 'conv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            h = chainer.functions.relu(bn(conv(h)))
        return h


class Model(chainer.link.Chain):
    def __init__(self, convs: DeepConvolution1D, out_size: int):
        super().__init__()
        with self.init_scope():
            self.convs = convs
            self.last = chainer.links.ConvolutionND(1, convs.out_size, out_size, 1)

    def __call__(self, x):
        h = x
        h = self.convs(h)
        h = self.last(h)
        return h


def create(config: ModelConfig):
    convs = DeepConvolution1D(
        in_size=config.in_size,
        num_scale=config.num_scale,
        base_num_z=config.base_num_z,
    )
    model = Model(
        convs=convs,
        out_size=config.out_size,
    )
    return model
