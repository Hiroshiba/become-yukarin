import chainer


class DeepConvolution(chainer.link.Chain):
    def __init__(self, num_scale: int, base_num_z: int, **kwargs):
        super().__init__(**kwargs)
        self.num_scale = num_scale

        for i in range(num_scale):
            l = base_num_z * 2 ** i
            self.add_link('conv{}'.format(i + 1),
                          chainer.links.Convolution2D(None, l, 4, 2, 1, nobias=True))
            self.add_link('bn{}'.format(i + 1), chainer.links.BatchNormalization(l))

    def get_scaled_width(self, base_width):
        return base_width // (2 ** self.num_scale)

    def __call__(self, x):
        h = x
        for i in range(self.num_scale):
            conv = getattr(self, 'conv{}'.format(i + 1))
            bn = getattr(self, 'bn{}'.format(i + 1))
            chainer.functions.relu(bn(conv(h)))
        return h
